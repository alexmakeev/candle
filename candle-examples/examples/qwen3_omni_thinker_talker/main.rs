//! Thinker → Talker integration test
//!
//! Tests the full pipeline: Thinker hidden states → Talker codec tokens
//!
//! Usage:
//!   cargo run --example qwen3_omni_thinker_talker -- --model /path/to/qwen3-omni-q8_0.gguf

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use candle_transformers::models::quantized_qwen3_omni::{
    Gguf, Talker, TalkerConfig, Thinker, ThinkerConfig,
};
use clap::Parser;
use std::fs::File;
use std::io::BufReader;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: String,

    /// Run on CPU (default: use GPU if available)
    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Thinker -> Talker Integration Test ===\n");
    println!("Loading GGUF model from: {}", args.model);

    // Open GGUF file
    let file = File::open(&args.model).context("Failed to open GGUF file")?;
    let mut reader = BufReader::new(file);

    // Parse GGUF content
    let content = candle::quantized::gguf_file::Content::read(&mut reader)
        .context("Failed to parse GGUF file")?;

    println!("GGUF version: {:?}", content.magic);
    println!("Number of tensors: {}", content.tensor_infos.len());

    // Check for hidden_projection tensors
    println!("\nChecking for hidden_projection tensors:");
    let hp_tensors = [
        "talker.hidden_projection.linear_fc1.weight",
        "talker.hidden_projection.linear_fc1.bias",
        "talker.hidden_projection.linear_fc2.weight",
        "talker.hidden_projection.linear_fc2.bias",
    ];
    for name in &hp_tensors {
        let found = content.tensor_infos.contains_key(*name);
        println!("  {} - {}", name, if found { "FOUND" } else { "NOT FOUND" });
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

    // Load configs
    let thinker_cfg = ThinkerConfig::default();
    let talker_cfg = TalkerConfig::default();

    println!("\nThinker config:");
    println!("  hidden_size: {}", thinker_cfg.hidden_size);
    println!("  num_hidden_layers: {}", thinker_cfg.num_hidden_layers);
    println!("  num_experts: {}", thinker_cfg.num_experts);

    println!("\nTalker config:");
    println!("  hidden_size: {}", talker_cfg.hidden_size);
    println!("  num_codebooks: {}", talker_cfg.num_codebooks);

    // Load Thinker
    println!("\nLoading Thinker...");
    let mut thinker = Thinker::from_gguf(&mut gg, &thinker_cfg, &device)?;
    println!("Thinker loaded successfully!");

    // Load Talker
    println!("\nLoading Talker...");
    let mut talker = Talker::from_gguf(&mut gg, &talker_cfg, &device)?;
    println!("Talker loaded successfully!");

    // Check all required components
    println!("\nTalker components:");
    println!("  Hidden projection: {}", talker.has_hidden_projection());
    println!("  Text projection: {}", talker.has_text_projection());
    println!("  TalkerModel (MoE decoder): {}", talker.has_talker_model());
    println!("  TalkerModel layers: {}", talker.talker_model_num_layers());
    println!("  codec_head: {}", talker.has_codec_head());

    if !talker.has_hidden_projection() {
        println!("ERROR: hidden_projection not found in GGUF file!");
        println!("Cannot test Thinker -> Talker integration.");
        return Ok(());
    }
    if !talker.has_talker_model() {
        println!("ERROR: TalkerModel not found in GGUF file!");
        println!("Ensure talker.model.* tensors are present.");
        return Ok(());
    }
    if !talker.has_codec_head() {
        println!("ERROR: codec_head not found in GGUF file!");
        println!("Ensure talker.codec_head.weight is present.");
        return Ok(());
    }

    // Create test input: simple token sequence
    // Using token IDs that should be valid in Qwen vocabulary
    println!("\n=== Running Thinker Forward Pass ===");
    let test_tokens: Vec<u32> = vec![
        // "Hello" as a simple test
        9707, 3520, // arbitrary tokens
    ];
    let num_tokens = test_tokens.len();
    let input = Tensor::from_slice(&test_tokens, (1, num_tokens), &device)?;
    println!("Input tokens: {:?}", test_tokens);
    println!("Input shape: {:?}", input.dims());

    // Run Thinker
    let thinker_output = thinker.forward_text_only(&input)?;
    println!("Thinker output:");
    println!("  text_logits shape: {:?}", thinker_output.text_logits.dims());
    println!("  hidden_states shape: {:?}", thinker_output.hidden_states.dims());

    // Verify hidden_states dimensions
    let (batch, seq, hidden_dim) = thinker_output.hidden_states.dims3()?;
    println!("\nHidden states dimensions:");
    println!("  batch: {}", batch);
    println!("  seq: {}", seq);
    println!("  hidden_dim: {}", hidden_dim);
    assert_eq!(hidden_dim, 2048, "Expected Thinker hidden_dim=2048");

    // Run Talker from hidden states
    println!("\n=== Running Talker Forward Pass ===");
    let codec_tokens = talker.forward_from_hidden(&thinker_output.hidden_states)?;
    println!("Codec tokens shape: {:?}", codec_tokens.dims());

    // Verify output dimensions
    let (cb_batch, cb_seq, num_codebooks) = codec_tokens.dims3()?;
    println!("\nCodec tokens dimensions:");
    println!("  batch: {}", cb_batch);
    println!("  seq: {}", cb_seq);
    println!("  num_codebooks: {}", num_codebooks);
    assert_eq!(num_codebooks, 15, "Expected 15 codebooks");

    // Print some codec tokens
    let tokens_cpu = codec_tokens.to_device(&Device::Cpu)?;
    let tokens_flat = tokens_cpu.flatten_all()?;
    let tokens_vec: Vec<u32> = tokens_flat.to_vec1()?;

    println!("\nCodec tokens (first position, all codebooks):");
    for (i, &t) in tokens_vec.iter().take(15).enumerate() {
        print!("cb{}: {} ", i, t);
    }
    println!();

    println!("\n=== Integration Test PASSED ===");
    println!("Successfully ran: Thinker.forward_text_only() -> Talker.forward_from_hidden()");

    Ok(())
}
