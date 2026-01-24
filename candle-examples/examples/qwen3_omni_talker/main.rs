//! Talker test: Load GGUF and test Talker code predictor
//!
//! This example loads the quantized Qwen3-Omni model and tests
//! the Talker code_predictor with dummy codec tokens.
//!
//! Usage:
//!   cargo run --example qwen3_omni_talker_test -- --model /path/to/qwen3-omni-q8_0.gguf

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use candle_transformers::models::quantized_qwen3_omni::{Gguf, Talker, TalkerConfig};
use clap::Parser;
use std::fs::File;
use std::io::BufReader;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: String,

    /// Number of input tokens per codebook
    #[arg(long, default_value = "10")]
    num_tokens: usize,

    /// Run on CPU (default: use GPU if available)
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading GGUF model from: {}", args.model);

    // Open GGUF file
    let file = File::open(&args.model).context("Failed to open GGUF file")?;
    let mut reader = BufReader::new(file);

    // Parse GGUF content
    let content = candle::quantized::gguf_file::Content::read(&mut reader)
        .context("Failed to parse GGUF file")?;

    println!("GGUF version: {:?}", content.magic);
    println!("Number of tensors: {}", content.tensor_infos.len());

    // List talker.code_predictor tensors
    println!("\nTalker code_predictor tensors found:");
    let mut talker_count = 0;
    for name in content.tensor_infos.keys() {
        if name.starts_with("talker.code_predictor.") {
            if talker_count < 15 {
                println!("  {}", name);
            }
            talker_count += 1;
        }
    }
    println!("  ... total {} talker.code_predictor tensors", talker_count);

    if talker_count == 0 {
        println!("\nNo talker.code_predictor tensors found. Checking for other talker tensors...");
        for name in content.tensor_infos.keys() {
            if name.starts_with("talker.") {
                println!("  {}", name);
            }
        }
        return Ok(());
    }

    // Select device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("\nUsing device: {:?}", device);

    // Create GGUF loader
    let mut gg = Gguf::new(content, reader, device.clone());

    // Load Talker config
    let cfg = TalkerConfig::default();
    println!("\nTalker config:");
    println!("  hidden_size: {}", cfg.hidden_size);
    println!("  num_attention_heads: {}", cfg.num_attention_heads);
    println!("  num_key_value_heads: {}", cfg.num_key_value_heads);
    println!("  head_dim: {}", cfg.head_dim);
    println!("  num_codebooks: {}", cfg.num_codebooks);
    println!("  codebook_size: {}", cfg.codebook_size);

    // Load Talker
    println!("\nLoading Talker...");
    let mut talker = Talker::from_gguf(&mut gg, &cfg, &device)?;
    println!("Talker loaded successfully!");

    // Create dummy tokens for testing
    // Shape: [batch=1, seq=num_tokens, num_codebooks=15]
    println!("\nRunning forward pass with {} dummy tokens...", args.num_tokens);

    let num_codebooks = talker.num_codebooks();
    let codebook_size = talker.codebook_size();

    // Create random tokens in valid range (0..codebook_size)
    let tokens: Vec<u32> = (0..args.num_tokens * num_codebooks)
        .map(|i| (i as u32 * 17) % codebook_size as u32)
        .collect();

    let token_tensor = Tensor::from_slice(&tokens, (1, args.num_tokens, num_codebooks), &device)?;
    println!("Input tokens shape: {:?}", token_tensor.dims());

    // Run forward pass
    let output = talker.forward(&token_tensor)?;
    println!("Output tokens shape: {:?}", output.dims());

    // Get output tokens
    let output_cpu = output.to_device(&Device::Cpu)?;
    let output_flat = output_cpu.flatten_all()?;
    let output_vec: Vec<u32> = output_flat.to_vec1()?;

    println!("\nOutput tokens (first 30):");
    for (i, &t) in output_vec.iter().take(30).enumerate() {
        print!("{}", t);
        if i < 29 {
            print!(", ");
        }
        if (i + 1) % 15 == 0 {
            println!();
        }
    }
    println!();

    // Test KV cache by running incremental generation
    println!("\nTesting incremental generation with KV cache...");
    talker.clear_kv_cache();

    // First step: full sequence
    let step1 = talker.forward(&token_tensor)?;
    println!("Step 1 output shape: {:?}", step1.dims());

    // Second step: single new token
    let single_tokens: Vec<u32> = (0..num_codebooks)
        .map(|i| (i as u32 * 31) % codebook_size as u32)
        .collect();
    let single_token_tensor = Tensor::from_slice(&single_tokens, (1, 1, num_codebooks), &device)?;
    let step2 = talker.forward(&single_token_tensor)?;
    println!("Step 2 output shape: {:?}", step2.dims());

    println!("\nSuccess! Talker code_predictor is working.");

    Ok(())
}
