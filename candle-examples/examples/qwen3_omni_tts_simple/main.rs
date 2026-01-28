//! Qwen3-Omni Simple TTS: Text -> Audio
//!
//! Complete TTS pipeline:
//! 1. Thinker: Text tokens -> Hidden states
//! 2. Talker: Hidden states -> Codec tokens (15 codebooks)
//! 3. Code2Wav: First 8 codebooks -> Audio waveform
//!
//! Usage:
//!   cargo run --release --example qwen3_omni_tts_simple -- \
//!     --model /path/to/qwen3-omni-q8_0.gguf \
//!     --output output.wav

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use candle_transformers::models::quantized_qwen3_omni::{
    Code2Wav, Code2WavConfig, Gguf, Talker, TalkerConfig, Thinker, ThinkerConfig,
};
use clap::Parser;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};

const SAMPLE_RATE: u32 = 24000;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3-Omni Simple TTS")]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: String,

    /// Output WAV file path
    #[arg(long, default_value = "output.wav")]
    output: String,

    /// Run on CPU (default: use GPU if available)
    #[arg(long)]
    cpu: bool,
}

/// Write audio samples to a WAV file
fn save_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2; // 16-bit mono
    let block_align = 2u16;
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    // RIFF header
    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;

    // fmt chunk
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?; // PCM
    writer.write_all(&1u16.to_le_bytes())?; // mono
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?; // bits per sample

    // data chunk
    writer.write_all(b"data")?;
    writer.write_all(&data_size.to_le_bytes())?;

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

    println!("=== Qwen3-Omni Simple TTS ===\n");

    // Select device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("Using device: {:?}", device);

    // Load GGUF
    println!("\nLoading GGUF: {}", args.model);
    let file = File::open(&args.model).context("Failed to open GGUF")?;
    let mut reader = BufReader::new(file);
    let content = candle::quantized::gguf_file::Content::read(&mut reader)?;
    println!("  Tensors: {}", content.tensor_infos.len());

    let mut gg = Gguf::new(content, reader, device.clone());

    // Load configs
    let thinker_cfg = ThinkerConfig::default();
    let talker_cfg = TalkerConfig::default();
    let code2wav_cfg = Code2WavConfig::default();

    println!("\nCode2Wav config:");
    println!("  num_codebooks: {}", code2wav_cfg.num_codebooks);
    println!("  codebook_size: {}", code2wav_cfg.codebook_size);
    println!("  embedding vocab: {}", code2wav_cfg.num_codebooks * code2wav_cfg.codebook_size);

    // Load models
    println!("\nLoading Thinker...");
    let start = std::time::Instant::now();
    let mut thinker = Thinker::from_gguf(&mut gg, &thinker_cfg, &device)?;
    println!("  Done in {:.2}s", start.elapsed().as_secs_f64());

    println!("\nLoading Talker...");
    let start = std::time::Instant::now();
    let mut talker = Talker::from_gguf(&mut gg, &talker_cfg, &device)?;
    println!("  Done in {:.2}s", start.elapsed().as_secs_f64());

    if !talker.has_hidden_projection() {
        anyhow::bail!("Talker missing hidden_projection");
    }

    println!("\nLoading Code2Wav...");
    let start = std::time::Instant::now();
    let code2wav = Code2Wav::from_gguf(&mut gg, &code2wav_cfg, &device)?;
    println!("  Done in {:.2}s", start.elapsed().as_secs_f64());

    // Simple test tokens (arbitrary valid tokens)
    let test_tokens: Vec<u32> = vec![9707, 3520, 1234, 5678];
    let num_tokens = test_tokens.len();

    println!("\n=== TTS Pipeline ===\n");
    println!("Input tokens: {:?}", test_tokens);

    // Step 1: Thinker forward
    println!("\nStep 1: Thinker...");
    let input = Tensor::from_slice(&test_tokens, (1, num_tokens), &device)?;
    let start = std::time::Instant::now();
    let thinker_output = thinker.forward_text_only(&input)?;
    println!(
        "  Hidden states: {:?} ({:.2}s)",
        thinker_output.hidden_states.dims(),
        start.elapsed().as_secs_f64()
    );

    // Step 2: Talker forward
    println!("\nStep 2: Talker...");
    let start = std::time::Instant::now();
    let codec_tokens = talker.forward_from_hidden(&thinker_output.hidden_states)?;
    let (batch, seq_len, num_cb) = codec_tokens.dims3()?;
    println!(
        "  Codec tokens: [batch={}, seq={}, codebooks={}] ({:.2}s)",
        batch, seq_len, num_cb,
        start.elapsed().as_secs_f64()
    );

    // Debug: token range
    let tokens_cpu = codec_tokens.to_device(&Device::Cpu)?;
    let tokens_flat = tokens_cpu.flatten_all()?;
    let tokens_vec: Vec<u32> = tokens_flat.to_vec1()?;
    let max_token = tokens_vec.iter().max().unwrap_or(&0);
    let min_token = tokens_vec.iter().min().unwrap_or(&0);
    println!("  Token range: {} - {}", min_token, max_token);

    // Step 3: Use first 8 codebooks for Code2Wav
    // Talker outputs: [batch, seq, 15] with tokens 0-2047
    // Code2Wav expects: [batch, 8, seq] with tokens 0-4095
    // Since talker tokens < 4096, we can use them directly
    println!("\nStep 3: Code2Wav...");

    // Take first 8 codebooks
    let first_8_codebooks = codec_tokens.narrow(2, 0, 8)?;
    println!("  Using first 8 codebooks: {:?}", first_8_codebooks.dims());

    // Transpose to [batch, codebooks, seq]
    let codec_transposed = first_8_codebooks.transpose(1, 2)?;
    println!("  Transposed: {:?}", codec_transposed.dims());

    // Code2Wav forward with offset=0
    let start = std::time::Instant::now();
    let audio = code2wav.forward(&codec_transposed)?;
    println!(
        "  Audio: {:?} ({:.2}s)",
        audio.dims(),
        start.elapsed().as_secs_f64()
    );

    // Extract and save audio
    let audio_cpu = audio.to_device(&Device::Cpu)?.flatten_all()?;
    let samples: Vec<f32> = audio_cpu.to_vec1()?;

    let duration_sec = samples.len() as f64 / SAMPLE_RATE as f64;
    println!("\nAudio statistics:");
    println!("  Samples: {}", samples.len());
    println!("  Duration: {:.2}s", duration_sec);

    let min = samples.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = samples.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let rms: f32 = (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt();
    println!("  Range: [{:.4}, {:.4}]", min, max);
    println!("  RMS: {:.4}", rms);

    println!("\nSaving: {}", args.output);
    save_wav(&args.output, &samples, SAMPLE_RATE)?;

    println!("\n=== Done ===");
    println!("Output: {} ({:.2}s audio)", args.output, duration_sec);

    Ok(())
}
