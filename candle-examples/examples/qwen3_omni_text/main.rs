#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, IndexOp, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen3_omni::{Thinker, ThinkerConfig};
use tokenizers::Tokenizer;

struct TextGeneration {
    model: Thinker,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Thinker,
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

        println!("Prompt tokens: {} tokens", tokens.len());
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

        // Initial forward pass with full prompt
        let input = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let output = self.model.forward_text_only(&input)?;
        let logits = output
            .text_logits
            .i((0, output.text_logits.dim(1)? - 1))?
            .to_dtype(DType::F32)?;

        let mut next_token = self.logits_processor.sample(&logits)?;
        tokens.push(next_token);
        generated_tokens += 1;

        if next_token == eos_token || next_token == eos_token2 {
            let dt = start_gen.elapsed();
            println!(
                "\n{generated_tokens} tokens generated ({:.2} token/s)",
                generated_tokens as f64 / dt.as_secs_f64(),
            );
            return Ok(());
        }

        if let Some(t) = self.tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }

        // Autoregressive generation
        for _ in 1..sample_len {
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let output = self.model.forward_text_only(&input)?;
            let logits = output
                .text_logits
                .i((0, 0))?
                .to_dtype(DType::F32)?;

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

            let next_token_new = self.logits_processor.sample(&logits)?;
            tokens.push(next_token_new);
            generated_tokens += 1;

            if next_token_new == eos_token || next_token_new == eos_token2 {
                break;
            }

            if let Some(t) = self.tokenizer.next_token(next_token_new)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }

            next_token = next_token_new;
        }

        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
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

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

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
    #[arg(long, short = 'n', default_value_t = 100)]
    sample_len: usize,

    /// Path to model weights directory (containing safetensors files)
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
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );
    println!(
        "temp: {:.2} repeat-penalty: {:.2} repeat-last-n: {}",
        args.temperature.unwrap_or(0.),
        args.repeat_penalty,
        args.repeat_last_n
    );

    let start = std::time::Instant::now();

    // Load tokenizer
    let tokenizer_path = match args.tokenizer_file.as_ref() {
        Some(file) => std::path::PathBuf::from(file),
        None => std::path::Path::new(&args.weight_path).join("tokenizer.json"),
    };
    println!("Loading tokenizer from: {}", tokenizer_path.display());
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(E::msg)?;

    // Load config
    let config_path = std::path::Path::new(&args.weight_path).join("config.json");
    println!("Loading config from: {}", config_path.display());
    let config_json = std::fs::read(&config_path)?;
    let config: ThinkerConfig = serde_json::from_slice(&config_json)?;

    println!("Config: {:?}", config);
    println!("Retrieved files in {:?}", start.elapsed());

    // Setup device
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    };
    println!("Device: {:?}, dtype: {:?}", device, dtype);

    // Load model weights
    let start = std::time::Instant::now();
    let weight_path = std::path::Path::new(&args.weight_path);

    // Check for model.safetensors.index.json or individual files
    let filenames = if weight_path.join("model.safetensors.index.json").exists() {
        println!("Loading sharded model from index file");
        candle_examples::hub_load_local_safetensors(
            args.weight_path.clone(),
            "model.safetensors.index.json",
        )?
    } else if weight_path.join("model.safetensors").exists() {
        println!("Loading single model file");
        vec![weight_path.join("model.safetensors")]
    } else {
        // Look for model-*.safetensors files
        let mut files: Vec<_> = std::fs::read_dir(weight_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|s| s == "safetensors")
                    .unwrap_or(false)
                    && p.file_name()
                        .and_then(|s| s.to_str())
                        .map(|s| s.starts_with("model-"))
                        .unwrap_or(false)
            })
            .collect();
        files.sort();
        println!("Loading {} shard files", files.len());
        files
    };

    for (i, f) in filenames.iter().enumerate() {
        println!("  [{}] {}", i + 1, f.display());
    }

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
    println!("Loading Thinker model...");
    let vb_thinker = vb.pp("thinker");
    let model = Thinker::new(&config, vb_thinker)?;

    println!("Loaded the model in {:?}", start.elapsed());

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
