#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3::{Config, ModelForCausalLM};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: ModelForCausalLM,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: ModelForCausalLM,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        eprintln!("[GEN] Prompt tokens: {}", tokens.len());
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<|endoftext|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|endoftext|> token"),
        };
        let eos_token2 = match self.tokenizer.get_token("<|im_end|>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <|im_end|> token"),
        };

        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            eprintln!("[GEN] Forward pass {} (start_pos={}, ctx={})", index, start_pos, ctxt.len());
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eos_token2 {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        eprintln!(
            "\n[GEN] {} tokens generated ({:.2} token/s)",
            generated_tokens,
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    #[arg(long)]
    prompt: String,

    /// The temperature used to generate samples.
    #[arg(long)]
    temperature: Option<f64>,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, short = 'n', default_value_t = 20)]
    sample_len: usize,

    /// Path to model weights directory
    #[arg(long)]
    weight_path: String,

    /// Path to tokenizer.json file
    #[arg(long)]
    tokenizer_file: Option<String>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}

fn main() -> Result<()> {
    use std::io::Write;
    macro_rules! log {
        ($($arg:tt)*) => {{
            eprintln!($($arg)*);
            std::io::stderr().flush().ok();
        }};
    }

    let args = Args::parse();

    log!("[INIT] avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    // Load tokenizer
    let tokenizer_path = match args.tokenizer_file.as_ref() {
        Some(file) => std::path::PathBuf::from(file),
        None => std::path::Path::new(&args.weight_path).join("tokenizer.json"),
    };
    log!("[LOAD] Loading tokenizer from: {}", tokenizer_path.display());
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;
    log!("[LOAD] Tokenizer loaded OK");

    // Load config
    let config_path = std::path::Path::new(&args.weight_path).join("config.json");
    log!("[LOAD] Loading config from: {}", config_path.display());
    let config: Config = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    log!("[LOAD] Config: hidden_size={}, layers={}, heads={}, vocab={}",
        config.hidden_size, config.num_hidden_layers, config.num_attention_heads, config.vocab_size);

    // Setup device
    log!("[DEVICE] Setting up device...");
    #[cfg(feature = "wgpu")]
    let wgpu_available = if !args.cpu {
        log!("[WGPU] Checking wgpu availability...");
        if candle::wgpu_backend::is_available() {
            let adapters = candle::wgpu_backend::list_adapters();
            log!("[WGPU] Found {} adapter(s)", adapters.len());
            for (i, info) in adapters.iter().enumerate() {
                log!("[WGPU]   [{i}] {:?}", info);
            }
            true
        } else {
            log!("[WGPU] Not available");
            false
        }
    } else {
        false
    };
    #[cfg(not(feature = "wgpu"))]
    let wgpu_available = false;

    #[cfg(feature = "wgpu")]
    let device = if wgpu_available && !args.cpu {
        log!("[DEVICE] Creating Device::new_wgpu(0)...");
        Device::new_wgpu(0)?
    } else {
        candle_examples::device(args.cpu)?
    };
    #[cfg(not(feature = "wgpu"))]
    let device = candle_examples::device(args.cpu)?;

    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        #[cfg(feature = "wgpu")]
        {
            if device.is_wgpu() {
                log!("[DEVICE] Using BF16 on wgpu");
                DType::BF16
            } else {
                DType::F32
            }
        }
        #[cfg(not(feature = "wgpu"))]
        DType::F32
    };
    log!("[DEVICE] dtype: {:?}", dtype);

    // Load weights
    let start = std::time::Instant::now();
    let weight_path = std::path::Path::new(&args.weight_path);

    log!("[WEIGHTS] Resolving safetensors files...");
    let filenames = if weight_path.join("model.safetensors.index.json").exists() {
        log!("[WEIGHTS] Found sharded model");
        candle_examples::hub_load_local_safetensors(
            args.weight_path.clone(),
            "model.safetensors.index.json",
        )?
    } else if weight_path.join("model.safetensors").exists() {
        log!("[WEIGHTS] Found single model file");
        vec![weight_path.join("model.safetensors")]
    } else {
        anyhow::bail!("No safetensors files found in {}", weight_path.display());
    };
    log!("[WEIGHTS] {} file(s) to load", filenames.len());

    let model = {
        log!("[MMAP] Memory-mapping safetensors files (streaming)...");
        let vb = unsafe { VarBuilder::from_mmaped_safetensors_streaming(&filenames, dtype, &device)? };
        log!("[MMAP] VarBuilder created OK");
        log!("[MODEL] Creating Qwen3 model...");
        let model = ModelForCausalLM::new(&config, vb)?;
        log!("[MODEL] Model loaded, dropping VarBuilder...");
        model
    };
    log!("[MODEL] Loaded in {:?}", start.elapsed());

    let mut pipeline = TextGeneration::new(
        model,
        tokenizer,
        args.seed,
        args.temperature,
        args.top_p,
        args.repeat_penalty,
        args.repeat_last_n,
        &device,
    );
    pipeline.run(&args.prompt, args.sample_len)?;
    Ok(())
}
