//! Qwen3-Omni TTS with Text Input
//!
//! Complete TTS pipeline with tokenizer:
//! 1. Tokenize text input
//! 2. Thinker: Text tokens -> Hidden states
//! 3. Talker: Hidden states -> Codec tokens
//! 4. Code2Wav: First 8 codebooks -> Audio
//!
//! Usage:
//!   cargo run --release --example qwen3_omni_tts_text -- \
//!     --model /path/to/qwen3-omni-q8_0.gguf \
//!     --text "Hello, world!" \
//!     --output output.wav

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use candle_transformers::models::quantized_qwen3_omni::{
    Code2Wav, Code2WavConfig, Gguf, Talker, TalkerConfig, Thinker, ThinkerConfig,
};
use clap::Parser;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use tokenizers::Tokenizer;

const SAMPLE_RATE: u32 = 24000;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3-Omni TTS with Text")]
struct Args {
    /// Path to GGUF model file
    #[arg(long)]
    model: String,

    /// Path to tokenizer.json (downloads if not specified)
    #[arg(long)]
    tokenizer: Option<String>,

    /// Text to synthesize
    #[arg(long, default_value = "Hello, world!")]
    text: String,

    /// Output WAV file path
    #[arg(long, default_value = "output.wav")]
    output: String,

    /// Run on CPU (default: use GPU if available)
    #[arg(long)]
    cpu: bool,
}

fn save_wav(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let num_samples = samples.len() as u32;
    let byte_rate = sample_rate * 2;
    let block_align = 2u16;
    let data_size = num_samples * 2;
    let file_size = 36 + data_size;

    writer.write_all(b"RIFF")?;
    writer.write_all(&file_size.to_le_bytes())?;
    writer.write_all(b"WAVE")?;
    writer.write_all(b"fmt ")?;
    writer.write_all(&16u32.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&1u16.to_le_bytes())?;
    writer.write_all(&sample_rate.to_le_bytes())?;
    writer.write_all(&byte_rate.to_le_bytes())?;
    writer.write_all(&block_align.to_le_bytes())?;
    writer.write_all(&16u16.to_le_bytes())?;
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

    println!("=== Qwen3-Omni TTS ===\n");
    println!("Text: \"{}\"", args.text);

    // Load tokenizer
    println!("\nLoading tokenizer...");
    let tokenizer_path = match &args.tokenizer {
        Some(path) => std::path::PathBuf::from(path),
        None => {
            println!("  Downloading from HuggingFace Hub...");
            let api = hf_hub::api::sync::Api::new()?;
            let repo = "Qwen/Qwen3-30B-A3B-Instruct-2507";
            api.model(repo.to_string()).get("tokenizer.json")?
        }
    };
    println!("  Path: {}", tokenizer_path.display());

    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

    // Tokenize text with chat template
    let chat_text = format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.text
    );
    let encoding = tokenizer
        .encode(chat_text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    println!("\nTokenization:");
    println!("  Chat template applied");
    println!("  Tokens: {} (first 20: {:?})", token_ids.len(), &token_ids[..token_ids.len().min(20)]);

    // Select device
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::cuda_if_available(0)?
    };
    println!("\nUsing device: {:?}", device);

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

    // TTS Pipeline
    println!("\n=== TTS Pipeline ===\n");

    // Step 1: Thinker forward
    println!("Step 1: Thinker forward...");
    let input = Tensor::from_slice(&token_ids, (1, token_ids.len()), &device)?;
    let start = std::time::Instant::now();
    let thinker_output = thinker.forward_text_only(&input)?;
    let (_, seq_len, _) = thinker_output.hidden_states.dims3()?;
    println!(
        "  Hidden states: [1, {}, 2048] ({:.2}s)",
        seq_len,
        start.elapsed().as_secs_f64()
    );

    // Step 2: Talker forward (autoregressive generation)
    println!("\nStep 2: Talker autoregressive generation...");
    let max_audio_frames = 50; // ~2 seconds at 12.5 Hz frame rate (reduced for testing)
    let start = std::time::Instant::now();
    let codec_tokens = talker.forward_from_hidden_autoregressive(
        &thinker_output.hidden_states,
        max_audio_frames,
    )?;
    let (_, cb_seq, num_cb) = codec_tokens.dims3()?;
    println!(
        "  Codec tokens: [1, {}, {}] ({:.2}s)",
        cb_seq, num_cb,
        start.elapsed().as_secs_f64()
    );

    // Token range check
    let tokens_cpu = codec_tokens.to_device(&Device::Cpu)?;
    let tokens_flat = tokens_cpu.flatten_all()?;
    let tokens_vec: Vec<u32> = tokens_flat.to_vec1()?;
    let max_token = tokens_vec.iter().max().unwrap_or(&0);
    let min_token = tokens_vec.iter().min().unwrap_or(&0);
    println!("  Token range: {} - {}", min_token, max_token);

    // Step 3: Code2Wav
    println!("\nStep 3: Code2Wav synthesis...");

    // Use first 8 codebooks
    let first_8_codebooks = codec_tokens.narrow(2, 0, 8)?;
    let codec_transposed = first_8_codebooks.transpose(1, 2)?;
    println!("  Input: {:?}", codec_transposed.dims());

    let start = std::time::Instant::now();
    let audio = code2wav.forward(&codec_transposed)?;
    println!(
        "  Audio: {:?} ({:.2}s)",
        audio.dims(),
        start.elapsed().as_secs_f64()
    );

    // Extract and save
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
    println!("Text: \"{}\"", args.text);
    println!("Output: {} ({:.2}s audio)", args.output, duration_sec);

    Ok(())
}
