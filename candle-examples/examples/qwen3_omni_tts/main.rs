//! Qwen3-Omni TTS example: Load GGUF and test Code2Wav
//!
//! This example loads the quantized Qwen3-Omni model and tests
//! the Code2Wav vocoder with dummy codec tokens.
//!
//! Usage:
//!   cargo run --example qwen3_omni_tts -- --model /path/to/qwen3-omni-q8_0.gguf

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use candle_transformers::models::quantized_qwen3_omni::{Code2Wav, Code2WavConfig, Gguf};
use clap::Parser;
use std::fs::File;
use std::io::BufReader;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: String,

    /// Number of test tokens
    #[arg(long, default_value = "100")]
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
    println!("Metadata entries: {}", content.metadata.len());

    // List code2wav tensors
    println!("\nCode2Wav tensors found:");
    let mut code2wav_count = 0;
    for name in content.tensor_infos.keys() {
        if name.starts_with("code2wav.") {
            if code2wav_count < 10 {
                println!("  {}", name);
            }
            code2wav_count += 1;
        }
    }
    println!("  ... total {} code2wav tensors", code2wav_count);

    // Select device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("\nUsing device: {:?}", device);

    // Create GGUF loader
    let mut gg = Gguf::new(content, reader, device.clone());

    // Load Code2Wav config
    let cfg = Code2WavConfig::default();
    println!("\nCode2Wav config:");
    println!("  embedding_dim: {}", cfg.embedding_dim);
    println!("  num_transformer_layers: {}", cfg.num_transformer_layers);
    println!("  num_codebooks: {}", cfg.num_codebooks);
    println!("  codebook_size: {}", cfg.codebook_size);

    // Load Code2Wav
    println!("\nLoading Code2Wav...");
    let code2wav = Code2Wav::from_gguf(&mut gg, &cfg, &device)?;
    println!("Code2Wav loaded successfully!");

    // Create dummy tokens for testing
    println!("\nRunning forward pass with {} dummy tokens...", args.num_tokens);

    // Create random tokens in valid range (0..32768)
    let vocab_size = cfg.codebook_size * cfg.num_codebooks;
    let tokens: Vec<u32> = (0..args.num_tokens)
        .map(|i| (i as u32 * 17) % vocab_size as u32)
        .collect();

    let token_tensor = Tensor::from_slice(&tokens, (1, args.num_tokens), &device)?;
    println!("Input tokens shape: {:?}", token_tensor.dims());

    // Run forward pass
    let output = code2wav.forward(&token_tensor)?;
    println!("Output audio shape: {:?}", output.dims());

    // Calculate expected output length
    // 2x upsample + 4 decoder blocks (8, 5, 4, 3) = 2*2*8*5*4*3 = 1920x
    let expected_upsample = 2 * 2 * 8 * 5 * 4 * 3;
    let expected_samples = args.num_tokens * expected_upsample;
    println!("Expected samples (approx): {}", expected_samples);

    // Get some statistics about output
    let output_cpu = output.to_device(&Device::Cpu)?.flatten_all()?;
    let output_vec: Vec<f32> = output_cpu.to_vec1()?;

    let min = output_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = output_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mean: f32 = output_vec.iter().sum::<f32>() / output_vec.len() as f32;

    println!("\nOutput statistics:");
    println!("  samples: {}", output_vec.len());
    println!("  min: {:.4}", min);
    println!("  max: {:.4}", max);
    println!("  mean: {:.4}", mean);

    println!("\nSuccess! Code2Wav vocoder is working.");

    Ok(())
}
