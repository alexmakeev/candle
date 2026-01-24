//! Qwen3-Omni Text-Only Generation Test
//!
//! Tests ONLY text generation without TTS pipeline.
//! Verifies that Thinker can generate coherent text responses.
//!
//! Usage:
//!   cargo run --example qwen3_omni_text_only -- \
//!     --model /path/to/qwen3-omni-q8_0.gguf \
//!     --tokenizer /path/to/tokenizer.json

use anyhow::{Context, Result};
use candle::{Device, IndexOp, Tensor};
use candle_transformers::models::quantized_qwen3_omni::{Gguf, Thinker, ThinkerConfig};
use clap::Parser;
use std::fs::File;
use std::io::BufReader;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: String,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: String,

    /// Prompt for text generation
    #[arg(long, default_value = "Какая столица Франции?")]
    prompt: String,

    /// Number of tokens to generate
    #[arg(long, default_value_t = 20)]
    max_tokens: usize,

    /// Temperature for sampling
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Run on CPU (default: use GPU if available)
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Qwen3-Omni Text-Only Generation ===\n");
    println!("Model: {}", args.model);
    println!("Tokenizer: {}", args.tokenizer);
    println!("Prompt: {}", args.prompt);
    println!();

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&args.tokenizer)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Tokenize prompt
    let encoding = tokenizer
        .encode(args.prompt.clone(), false)
        .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;
    let input_tokens = encoding.get_ids().to_vec();
    println!("Input tokens ({} total): {:?}", input_tokens.len(), input_tokens);

    // Open GGUF file
    let file = File::open(&args.model).context("Failed to open GGUF file")?;
    let mut reader = BufReader::new(file);

    // Parse GGUF content
    let content = candle::quantized::gguf_file::Content::read(&mut reader)
        .context("Failed to parse GGUF file")?;

    println!("GGUF loaded: {} tensors", content.tensor_infos.len());

    // Select device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("Using device: {:?}\n", device);

    // Create GGUF loader
    let mut gg = Gguf::new(content, reader, device.clone());

    // Load Thinker
    let thinker_cfg = ThinkerConfig::default();
    println!("Loading Thinker (hidden_size={})...", thinker_cfg.hidden_size);
    let mut thinker = Thinker::from_gguf(&mut gg, &thinker_cfg, &device)?;
    println!("Thinker loaded!\n");

    // First forward pass with full prompt
    let input = Tensor::from_slice(&input_tokens, (1, input_tokens.len()), &device)?;
    let output = thinker.forward_text_only(&input)?;

    // Get logits for last token
    let logits = output.text_logits;
    let (_, seq_len, _vocab_size) = logits.dims3()?;
    let last_logits = logits.i((0, seq_len - 1))?;

    // Apply temperature and get first token
    let scaled_logits = if args.temperature > 0.0 {
        (&last_logits / args.temperature)?
    } else {
        last_logits.clone()
    };
    let mut next_token = scaled_logits.argmax(0)?.to_vec0::<u32>()?;

    let mut generated_tokens = Vec::new();
    generated_tokens.push(next_token);

    println!("=== Generating Text ===\n");
    print!("Response: ");

    // Decode and print first token
    let token_str = tokenizer
        .decode(&[next_token], false)
        .map_err(|e| anyhow::anyhow!("Failed to decode token: {}", e))?;
    print!("{}", token_str);
    std::io::Write::flush(&mut std::io::stdout())?;

    // Continue generation using KV cache
    let mut offset = input_tokens.len();
    for _step in 1..args.max_tokens {
        // Stop on EOS token
        if next_token == 151643 {
            println!("\n[EOS reached]");
            break;
        }

        // Generate next token using KV cache
        let token_tensor = Tensor::from_slice(&[next_token], (1, 1), &device)?;
        let logits = thinker.forward_one_token(&token_tensor, offset)?;

        // Get logits and apply temperature
        let last_logits = logits.i((0, 0))?;
        let scaled_logits = if args.temperature > 0.0 {
            (&last_logits / args.temperature)?
        } else {
            last_logits.clone()
        };

        next_token = scaled_logits.argmax(0)?.to_vec0::<u32>()?;

        // Decode and print token
        let token_str = tokenizer
            .decode(&[next_token], false)
            .map_err(|e| anyhow::anyhow!("Failed to decode token: {}", e))?;
        print!("{}", token_str);
        std::io::Write::flush(&mut std::io::stdout())?;

        generated_tokens.push(next_token);
        offset += 1;
    }

    println!("\n\n=== Generation Complete ===");
    println!("Generated {} tokens", generated_tokens.len());
    println!("\nFull response:");
    let full_text = tokenizer
        .decode(&generated_tokens, false)
        .map_err(|e| anyhow::anyhow!("Failed to decode: {}", e))?;
    println!("{}", full_text);

    Ok(())
}
