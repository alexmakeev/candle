//! Qwen3-Omni ASR: Audio to Text

use anyhow::{Context, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::quantized_qwen3_omni::{
    AudioTower, AudioTowerConfig, Gguf, Thinker, ThinkerConfig,
};
use clap::Parser;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use tokenizers::Tokenizer;

const IM_START: u32 = 151644;
const IM_END: u32 = 151645;
const AUDIO_START: u32 = 151669;
const AUDIO_END: u32 = 151670;
const EOS: u32 = 151643;

// Mel filterbank for 128 mels, n_fft=400
const MEL_FILTERS: &[u8] = include_bytes!("whisper/melfilters128.bytes");

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3-Omni ASR")]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long)]
    tokenizer: Option<String>,
    #[arg(long)]
    audio: String,
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    prompt: Option<String>,
}

fn load_wav(path: &str) -> Result<(Vec<f32>, u32)> {
    let mut file = File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        anyhow::bail!("Invalid WAV");
    }

    let mut pos = 12;
    let (mut sr, mut bps, mut ch) = (0u32, 0u16, 0u16);

    while pos < data.len() - 8 {
        let id = &data[pos..pos + 4];
        let size = u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]) as usize;

        if id == b"fmt " {
            ch = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            sr = u32::from_le_bytes([data[pos + 12], data[pos + 13], data[pos + 14], data[pos + 15]]);
            bps = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);
        } else if id == b"data" {
            let audio = &data[pos + 8..pos + 8 + size];
            if bps != 16 { anyhow::bail!("Unsupported bits: {}", bps); }
            let samples: Vec<f32> = audio.chunks(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect();
            let mono = if ch == 2 { samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect() } else { samples };
            return Ok((mono, sr));
        }
        pos += 8 + size + (size % 2);
    }
    anyhow::bail!("No data chunk")
}

fn resample(s: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to { return s.to_vec(); }
    let r = from as f64 / to as f64;
    let n = (s.len() as f64 / r) as usize;
    (0..n).map(|i| {
        let src = i as f64 * r;
        let f = src.floor() as usize;
        let frac = (src - f as f64) as f32;
        if f + 1 < s.len() { s[f] * (1.0 - frac) + s[f + 1] * frac } else { s[f.min(s.len() - 1)] }
    }).collect()
}

fn compute_mel(samples: &[f32], device: &Device) -> Result<Tensor> {
    // Load mel filterbank from bytes
    let filters: Vec<f32> = MEL_FILTERS
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    
    let (n_fft, hop, n_mels) = (400, 160, 128);
    
    // Use whisper mel spectrogram
    let mel = candle_transformers::models::whisper::audio::log_mel_spectrogram_(
        samples, &filters, n_fft, hop, n_mels, false,
    );

    let frames = mel.len() / n_mels;
    println!("  Mel frames: {}, total values: {}", frames, mel.len());
    
    // Check mel stats
    let min = mel.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = mel.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    println!("  Mel range: [{:.3}, {:.3}]", min, max);

    // Shape: [1, 1, n_mels, frames] for Conv2d
    let mel_tensor = Tensor::from_slice(&mel, (n_mels, frames), device)?;
    Ok(mel_tensor.unsqueeze(0)?.unsqueeze(0)?.to_dtype(DType::F32)?)
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("=== Qwen3-Omni ASR ===");

    // Tokenizer
    let tok_path = match &args.tokenizer {
        Some(p) => std::path::PathBuf::from(p),
        None => hf_hub::api::sync::Api::new()?.model("Qwen/Qwen3-30B-A3B-Instruct-2507".to_string()).get("tokenizer.json")?,
    };
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Audio
    println!("Loading: {}", args.audio);
    let (samples, sr) = load_wav(&args.audio)?;
    println!("  Original: {} samples, {} Hz, {:.2}s", samples.len(), sr, samples.len() as f64 / sr as f64);
    
    let samples_16k = resample(&samples, sr, 16000);
    println!("  Resampled: {} samples at 16kHz", samples_16k.len());

    let device = if args.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    println!("Device: {:?}", device);

    // Mel
    println!("\nComputing mel spectrogram...");
    let mel = compute_mel(&samples_16k, &device)?;
    println!("  Tensor: {:?}", mel.dims());

    // GGUF
    println!("\nLoading model...");
    let file = File::open(&args.model).context("GGUF open failed")?;
    let mut reader = BufReader::new(file);
    let content = candle::quantized::gguf_file::Content::read(&mut reader)?;
    let mut gg = Gguf::new(content, reader, device.clone());

    let audio_tower = AudioTower::from_gguf(&mut gg, &AudioTowerConfig::default(), &device)?;
    let mut thinker = Thinker::from_gguf(&mut gg, &ThinkerConfig::default(), &device)?;

    // Process
    println!("\n=== Processing ===");
    let audio_embeds = audio_tower.forward(&mel)?;
    println!("Audio embeddings: {:?}", audio_embeds.dims());

    // Build prompt
    let sys = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.";
    let prompt = args.prompt.as_deref().unwrap_or("Transcribe this audio in original language.");

    // Tokenize
    let sys_enc = tokenizer.encode(format!("system\n{}", sys), false).map_err(|e| anyhow::anyhow!("{}", e))?;
    let user_pre = tokenizer.encode("user\n", false).map_err(|e| anyhow::anyhow!("{}", e))?;
    let user_post = tokenizer.encode(prompt, false).map_err(|e| anyhow::anyhow!("{}", e))?;
    let asst = tokenizer.encode("assistant\n", false).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Before audio: <|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n<|audio_start|>
    let mut before: Vec<u32> = vec![IM_START];
    before.extend(sys_enc.get_ids());
    before.push(IM_END);
    before.push(198);
    before.push(IM_START);
    before.extend(user_pre.get_ids());
    before.push(AUDIO_START);

    // After audio: <|audio_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n
    let mut after: Vec<u32> = vec![AUDIO_END];
    after.extend(user_post.get_ids());
    after.push(IM_END);
    after.push(198);
    after.push(IM_START);
    after.extend(asst.get_ids());

    println!("Tokens before audio: {}", before.len());
    println!("Audio seq length: {}", audio_embeds.dim(1)?);
    println!("Tokens after audio: {}", after.len());

    // Get text embeddings
    let before_t = Tensor::from_slice(&before, (1, before.len()), &device)?;
    let after_t = Tensor::from_slice(&after, (1, after.len()), &device)?;

    let before_emb = thinker.embed_tokens(&before_t)?;
    let after_emb = thinker.embed_tokens(&after_t)?;

    // Concatenate: [before_emb, audio_embeds, after_emb]
    let combined = Tensor::cat(&[&before_emb, &audio_embeds, &after_emb], 1)?;
    println!("Combined embeddings: {:?}", combined.dims());

    // Forward
    let output = thinker.forward_embeddings(&combined)?;
    let (_, seq_len, _) = output.text_logits.dims3()?;

    // Decode
    println!("\n=== Recognized Text ===");
    
    let last_logits = output.text_logits.i((0, seq_len - 1))?;
    let mut next_token = last_logits.argmax(0)?.to_vec0::<u32>()?;
    let mut generated = vec![next_token];

    print!("{}", tokenizer.decode(&[next_token], false).unwrap_or_default());
    std::io::stdout().flush()?;

    let mut offset = seq_len;
    for _ in 1..args.max_tokens {
        if next_token == EOS || next_token == IM_END { break; }
        let t = Tensor::from_slice(&[next_token], (1, 1), &device)?;
        let logits = thinker.forward_one_token(&t, offset)?;
        next_token = logits.i((0, 0))?.argmax(0)?.to_vec0::<u32>()?;
        generated.push(next_token);
        print!("{}", tokenizer.decode(&[next_token], false).unwrap_or_default());
        std::io::stdout().flush()?;
        offset += 1;
    }

    println!("\n\n=== Full Output ===");
    println!("{}", tokenizer.decode(&generated, true).unwrap_or_default());
    Ok(())
}
