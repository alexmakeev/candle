//! Qwen3-Omni Full TTS Pipeline: Text -> Audio
//!
//! This example runs the complete TTS pipeline:
//! 1. Tokenize text ("Privet mir" / "Hello world")
//! 2. Thinker: Text tokens -> Hidden states
//! 3. Talker: Hidden states -> Codec tokens (autoregressive)
//! 4. Code2Wav: Codec tokens -> Audio waveform
//! 5. Save as WAV file (24kHz mono)
//!
//! Usage:
//!   cargo run --release --example qwen3_omni_tts_full -- \
//!     --model /path/to/qwen3-omni-q8_0.gguf \
//!     --tokenizer /path/to/tokenizer.json \
//!     --prompt "Hello, world!" \
//!     --output output.wav

use anyhow::{Context, Result};
use candle::{DType, Device, Tensor};
use candle_transformers::models::quantized_qwen3_omni::{
    Code2Wav, Code2WavConfig, Gguf, Speaker, Talker, TalkerConfig, Thinker, ThinkerConfig,
};
use clap::Parser;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use tokenizers::Tokenizer;

const SAMPLE_RATE: u32 = 24000;

// TTS special token IDs from Thinker embed_tokens
const TTS_BOS_TOKEN_ID: u32 = 151672;
const TTS_EOS_TOKEN_ID: u32 = 151673;
const TTS_PAD_TOKEN_ID: u32 = 151671;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3-Omni TTS: Text to Speech")]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: String,

    /// Path to tokenizer.json
    #[arg(long)]
    tokenizer: Option<String>,

    /// Text prompt to synthesize
    #[arg(long, default_value = "Hello, world!")]
    prompt: String,

    /// Output WAV file path
    #[arg(long, default_value = "output.wav")]
    output: String,

    /// Run on CPU (default: use GPU if available)
    #[arg(long)]
    cpu: bool,

    /// Maximum generation steps (codec tokens)
    #[arg(long, default_value_t = 500)]
    max_steps: usize,

    /// Speaker voice (ethan, chelsie, aiden)
    #[arg(long, default_value = "ethan")]
    speaker: String,
}

/// Write audio samples to a WAV file
fn save_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // WAV header
    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono
    let block_align = 2u16; // 16-bit mono
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?; // chunk size
    writer.write_all(&1u16.to_le_bytes())?; // PCM format
    writer.write_all(&1u16.to_le_bytes())?; // mono
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

    // Convert f32 samples to i16
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let i16_sample = (clamped * 32767.0) as i16;
        writer.write_all(&i16_sample.to_le_bytes())?;
    }

    writer.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("=== Qwen3-Omni TTS Full Pipeline ===\n");

    // Load tokenizer
    println!("Loading tokenizer...");
    let tokenizer_path = match &args.tokenizer {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            // Try to download from HuggingFace Hub
            println!("  Downloading tokenizer from HuggingFace Hub...");
            let api = hf_hub::api::sync::Api::new()?;
            let repo = "Qwen/Qwen3-30B-A3B-Instruct-2507";
            api.model(repo.to_string()).get("tokenizer.json")?
        }
    };
    println!("  Path: {}", tokenizer_path.display());

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Tokenize input with chat template
    // Format: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    println!("\nTokenizing with chat template: \"{}\"", args.prompt);
    let chat_text = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.prompt
    );
    let encoding = tokenizer
        .encode(chat_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("  Chat text: {}", chat_text.replace('\n', "\\n"));
    println!("  Tokens: {:?}", &token_ids[..token_ids.len().min(20)]);
    println!("  Count: {}", token_ids.len());

    // Select device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("\nUsing device: {:?}", device);

    // Load GGUF model
    println!("\nLoading GGUF model: {}", args.model);
    let file = File::open(&args.model).context("Failed to open GGUF file")?;
    let mut reader = BufReader::new(file);

    let content = candle::quantized::gguf_file::Content::read(&mut reader)
        .context("Failed to parse GGUF file")?;

    println!("  Tensors: {}", content.tensor_infos.len());

    let mut gg = Gguf::new(content, reader, device.clone());

    // Load configs
    let thinker_cfg = ThinkerConfig::default();
    let talker_cfg = TalkerConfig::default();
    let code2wav_cfg = Code2WavConfig::default();

    // Load components
    println!("\nLoading Thinker (48 MoE layers)...");
    let start = std::time::Instant::now();
    let mut thinker = Thinker::from_gguf(&mut gg, &thinker_cfg, &device)?;
    println!("  Done in {:.2}s", start.elapsed().as_secs_f64());

    println!("\nLoading Talker...");
    let start = std::time::Instant::now();
    let mut talker = Talker::from_gguf(&mut gg, &talker_cfg, &device)?;
    println!("  Done in {:.2}s", start.elapsed().as_secs_f64());

    if !talker.has_hidden_projection() {
        anyhow::bail!("Talker missing hidden_projection - cannot connect to Thinker");
    }
    if !talker.has_text_projection() {
        anyhow::bail!("Talker missing text_projection - needed for proper TTS initialization");
    }

    // Parse speaker
    let speaker = match args.speaker.to_lowercase().as_str() {
        "ethan" => Speaker::Ethan,
        "chelsie" => Speaker::Chelsie,
        "aiden" => Speaker::Aiden,
        _ => anyhow::bail!("Unknown speaker: {}. Use ethan, chelsie, or aiden", args.speaker),
    };
    println!("  Speaker: {:?}", speaker);

    println!("\nLoading Code2Wav vocoder...");
    let start = std::time::Instant::now();
    let code2wav = Code2Wav::from_gguf(&mut gg, &code2wav_cfg, &device)?;
    println!("  Done in {:.2}s", start.elapsed().as_secs_f64());

    // Run pipeline
    println!("\n=== Running TTS Pipeline ===\n");

    // Step 1: Thinker forward pass to get embeddings
    println!("Step 1: Thinker forward pass...");
    let input_tensor = Tensor::from_slice(&token_ids, (1, token_ids.len()), &device)?;
    let start = std::time::Instant::now();
    let thinker_output = thinker.forward_text_only(&input_tensor)?;
    println!(
        "  Hidden states: {:?} ({:.2}s)",
        thinker_output.hidden_states.dims(),
        start.elapsed().as_secs_f64()
    );

    // Get TTS special token embeddings from Thinker embed_tokens
    // These are tokens: tts_bos (151672), tts_eos (151673), tts_pad (151671)
    println!("\nStep 1b: Getting TTS special token embeddings...");
    let tts_special_token_ids = Tensor::from_slice(
        &[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID],
        (1, 3),
        &device,
    )?;
    // Get raw embeddings from Thinker's embed_tokens layer
    let tts_special_embeds = thinker.embed_tokens(&tts_special_token_ids)?.squeeze(0)?; // [3, 2048]
    println!("  TTS special embeds: {:?}", tts_special_embeds.dims());

    // Step 2: Talker codec generation with proper speaker initialization
    println!("\nStep 2: Talker codec generation with speaker {:?}...", speaker);
    let start = std::time::Instant::now();

    let codec_tokens = talker.generate_with_speaker(
        &thinker_output.hidden_states,
        &tts_special_embeds,
        speaker,
        args.max_steps,
    )?;

    let (_batch, seq_len, num_codebooks) = codec_tokens.dims3()?;
    println!(
        "  Codec tokens: {:?} ({:.2}s)",
        codec_tokens.dims(),
        start.elapsed().as_secs_f64()
    );
    println!(
        "  Generated {} positions x {} codebooks",
        seq_len, num_codebooks
    );

    // Debug: print first few tokens
    let codec_cpu = codec_tokens.to_device(&Device::Cpu)?;
    println!("\n  First 5 positions of codec tokens:");
    for pos in 0..5.min(seq_len) {
        let mut tokens_str = String::new();
        for cb in 0..num_codebooks.min(8) {
            let t: u32 = codec_cpu.get(0)?.get(pos)?.get(cb)?.to_scalar()?;
            tokens_str.push_str(&format!("{} ", t));
        }
        println!("    pos {}: {}", pos, tokens_str);
    }

    // Debug: check token range
    let codec_flat = codec_cpu.flatten_all()?;
    let codec_vec: Vec<u32> = codec_flat.to_vec1()?;
    let max_token = codec_vec.iter().max().unwrap_or(&0);
    let min_token = codec_vec.iter().min().unwrap_or(&0);
    println!("  Token range: {} - {}", min_token, max_token);
    let over_2048: Vec<_> = codec_vec.iter().filter(|&&t| t >= 2048).collect();
    println!("  Tokens >= 2048: {} out of {}", over_2048.len(), codec_vec.len());
    if !over_2048.is_empty() && over_2048.len() <= 10 {
        println!("  Over-threshold values: {:?}", over_2048);
    }

    // Transpose from [batch, seq, codebooks] to [batch, codebooks, seq] for Code2Wav
    // Talker generates codebooks 0-14, which map to Code2Wav codebooks 1-15
    // Use forward_with_offset(tokens, 1) to add correct offset
    let codec_transposed = codec_tokens.transpose(1, 2)?; // [1, 15, seq]
    println!("  Codec for vocoder: {:?}", codec_transposed.dims());
    
    // Debug: check range in transposed tensor
    let trans_flat = codec_transposed.flatten_all()?;
    let trans_vec: Vec<u32> = trans_flat.to_vec1()?;
    let trans_max = trans_vec.iter().max().unwrap_or(&0);
    let trans_min = trans_vec.iter().min().unwrap_or(&0);
    println!("  Transposed range: {} - {}", trans_min, trans_max);

    // Step 3: Code2Wav synthesis (with offset 1 for Talker codebooks)
    // Talker outputs codebooks 0-14, Code2Wav expects 1-15
    println!("\nStep 3: Code2Wav audio synthesis...");
    let start = std::time::Instant::now();
    let audio = code2wav.forward_with_offset(&codec_transposed, 1)?;
    println!(
        "  Audio shape: {:?} ({:.2}s)",
        audio.dims(),
        start.elapsed().as_secs_f64()
    );

    // Extract audio samples
    let audio_cpu = audio.to_device(&Device::Cpu)?.flatten_all()?;
    let samples: Vec<f32> = audio_cpu.to_vec1()?;

    let duration_sec = samples.len() as f64 / SAMPLE_RATE as f64;
    println!("\nAudio statistics:");
    println!("  Samples: {}", samples.len());
    println!("  Duration: {:.2}s", duration_sec);
    println!("  Sample rate: {} Hz", SAMPLE_RATE);

    let min = samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let rms: f32 = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    println!("  Range: [{:.4}, {:.4}]", min, max);
    println!("  RMS: {:.4}", rms);

    // Save WAV
    println!("\nSaving to: {}", args.output);
    save_wav(&args.output, &samples, SAMPLE_RATE)?;
    println!("  Done!");

    println!("\n=== TTS Pipeline Complete ===");
    println!("Output: {} ({:.2}s of audio)", args.output, duration_sec);

    Ok(())
}
