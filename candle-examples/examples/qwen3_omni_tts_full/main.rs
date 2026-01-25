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
    Code2Wav, Code2WavConfig, Gguf, Talker, TalkerConfig, Thinker, ThinkerConfig,
};
use clap::Parser;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use tokenizers::Tokenizer;

const SAMPLE_RATE: u32 = 24000;
const STOP_TOKEN: u32 = 2150; // Stop token for codec generation (from Qwen3-Omni config)

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
    /// Default: auto-calculate based on input length
    #[arg(long)]
    max_steps: Option<usize>,
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

    // Tokenize input
    println!("\nTokenizing: \"{}\"", args.prompt);
    let encoding = tokenizer
        .encode(args.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("  Tokens: {:?}", token_ids);
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

    println!("\nLoading Code2Wav vocoder...");
    let start = std::time::Instant::now();
    let code2wav = Code2Wav::from_gguf(&mut gg, &code2wav_cfg, &device)?;
    println!("  Done in {:.2}s", start.elapsed().as_secs_f64());

    // Run pipeline
    println!("\n=== Running TTS Pipeline ===\n");

    // Step 1: Thinker forward
    println!("Step 1: Thinker forward pass...");
    let input_tensor = Tensor::from_slice(&token_ids, (1, token_ids.len()), &device)?;
    let start = std::time::Instant::now();
    let thinker_output = thinker.forward_text_only(&input_tensor)?;
    println!(
        "  Hidden states: {:?} ({:.2}s)",
        thinker_output.hidden_states.dims(),
        start.elapsed().as_secs_f64()
    );

    // Step 2: Talker autoregressive codec generation
    println!("\nStep 2: Talker codec generation (autoregressive)...");
    let start = std::time::Instant::now();

    // Get initial codec tokens from hidden states
    let initial_tokens = talker.forward_from_hidden(&thinker_output.hidden_states)?;
    let (_batch, _init_seq, num_codebooks) = initial_tokens.dims3()?;
    println!("  Initial tokens: {:?}", initial_tokens.dims());

    // Calculate max_steps based on input length if not provided
    // Heuristic: ~2-3 seconds per word at normal speech rate
    // Empirically: 3 codec frames per input token produces ~1.5s per word
    let max_steps = args.max_steps.unwrap_or_else(|| {
        let estimated = token_ids.len() * 3;
        println!("  Auto max_steps: {} ({}x input tokens)", estimated, 3);
        estimated.max(5) // Minimum 5 frames
    });

    // Autoregressive generation loop
    let mut all_tokens = vec![initial_tokens.clone()];
    let mut current_tokens = initial_tokens;

    // Clear KV cache and feed initial tokens
    talker.clear_kv_cache();

    for step in 0..max_steps {
        // Generate next token
        let next_logits = talker.forward_logits(&current_tokens)?;
        let (_b, seq, _cb, _vocab) = next_logits.dims4()?;

        // Take last position logits
        let last_logits = next_logits.narrow(1, seq - 1, 1)?;

        // Greedy decode for each codebook
        let next_tokens = last_logits.argmax(3)?.to_dtype(DType::U32)?;

        // Check for STOP token (token 2150 in any codebook)
        let tokens_cpu = next_tokens.to_device(&Device::Cpu)?.flatten_all()?;
        let tokens_vec: Vec<u32> = tokens_cpu.to_vec1()?;
        if tokens_vec.iter().any(|&t| t == STOP_TOKEN) {
            println!("  STOP token detected at step {}", step);
            break;
        }

        all_tokens.push(next_tokens.clone());
        current_tokens = next_tokens;

        if (step + 1) % 50 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout())?;
        }
    }
    println!();

    // Concatenate all tokens
    let codec_tokens = Tensor::cat(&all_tokens, 1)?;
    let (batch, seq_len, _) = codec_tokens.dims3()?;
    println!(
        "  Codec tokens: {:?} ({:.2}s)",
        codec_tokens.dims(),
        start.elapsed().as_secs_f64()
    );
    println!(
        "  Generated {} positions x {} codebooks",
        seq_len, num_codebooks
    );

    // Convert multi-codebook tokens to flat format for Code2Wav
    // Code2Wav uses 8 codebooks, but Talker outputs 15 codebooks
    // We use the first 8 codebooks from Talker output
    let code2wav_num_codebooks = 8;
    let code2wav_codebook_size = 4096u32; // From Code2WavConfig

    let mut flat_tokens: Vec<u32> = Vec::with_capacity(seq_len * code2wav_num_codebooks);

    let codec_cpu = codec_tokens.to_device(&Device::Cpu)?;
    for pos in 0..seq_len {
        for cb in 0..code2wav_num_codebooks {
            let token: u32 = codec_cpu.get(0)?.get(pos)?.get(cb)?.to_scalar()?;
            // Add codebook offset for Code2Wav embedding
            flat_tokens.push(token + cb as u32 * code2wav_codebook_size);
        }
    }
    // DEBUG: Print token statistics
    println!("\nDEBUG: flat_tokens stats:");
    let flat_min = flat_tokens.iter().min().unwrap();
    let flat_max = flat_tokens.iter().max().unwrap();
    println!("  Count: {}", flat_tokens.len());
    println!("  Range: [{}, {}]", flat_min, flat_max);
    println!("  Expected: [0, {}]", code2wav_num_codebooks as u32 * code2wav_codebook_size - 1);

    let codec_flat = Tensor::from_slice(&flat_tokens, (batch, seq_len * code2wav_num_codebooks), &device)?;

    // Step 3: Code2Wav synthesis
    println!("\nStep 3: Code2Wav audio synthesis...");
    let start = std::time::Instant::now();
    let audio = code2wav.forward(&codec_flat)?;
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
