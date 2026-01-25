//! Qwen3-Omni ASR using HuggingFace mel spectrogram

use anyhow::{Context, Result};
use candle::{DType, Device, IndexOp, Tensor};
use candle_transformers::models::quantized_qwen3_omni::{
    AudioTower, AudioTowerConfig, Gguf, Thinker, ThinkerConfig,
};
use clap::Parser;
use std::fs::File;
use std::io::{BufReader, Write};
use tokenizers::Tokenizer;

const IM_START: u32 = 151644;
const IM_END: u32 = 151645;
const AUDIO_START: u32 = 151669;
const AUDIO_END: u32 = 151670;
const EOS: u32 = 151643;

#[derive(Parser, Debug)]
#[command(author, version, about = "Qwen3-Omni ASR with HF mel")]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long)]
    tokenizer: Option<String>,
    #[arg(long)]
    mel: String,  // Path to mel .npy file
    #[arg(long, default_value_t = 100)]
    max_tokens: usize,
    #[arg(long)]
    cpu: bool,
    #[arg(long)]
    prompt: Option<String>,
}

fn load_npy(path: &str) -> Result<(Vec<f32>, usize, usize)> {
    let data = std::fs::read(path)?;
    
    // Simple NPY parser for f32 arrays
    if &data[0..6] != b"\x93NUMPY" {
        anyhow::bail!("Not a valid NPY file");
    }
    
    let version = (data[6], data[7]);
    let header_len = if version.0 == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    
    let header_start = if version.0 == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&data[header_start..header_start + header_len])?;
    
    // Parse shape from header like "{'shape': (128, 3000), ...}"
    let shape_start = header.find("'shape':").unwrap() + 8;
    let shape_str = &header[shape_start..];
    let paren_start = shape_str.find('(').unwrap() + 1;
    let paren_end = shape_str.find(')').unwrap();
    let dims: Vec<usize> = shape_str[paren_start..paren_end]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse().unwrap())
        .collect();
    
    let data_start = header_start + header_len;
    let floats: Vec<f32> = data[data_start..]
        .chunks(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    
    Ok((floats, dims[0], dims[1]))
}

fn main() -> Result<()> {
    let args = Args::parse();
    println!("=== Qwen3-Omni ASR (HF Mel) ===");

    // Tokenizer
    let tok_path = match &args.tokenizer {
        Some(p) => std::path::PathBuf::from(p),
        None => hf_hub::api::sync::Api::new()?.model("Qwen/Qwen3-30B-A3B-Instruct-2507".to_string()).get("tokenizer.json")?,
    };
    let tokenizer = Tokenizer::from_file(&tok_path).map_err(|e| anyhow::anyhow!("{}", e))?;

    // Load mel from NPY
    println!("Loading mel: {}", args.mel);
    let (mel_data, n_mels, n_frames) = load_npy(&args.mel)?;
    println!("  Shape: [{}, {}]", n_mels, n_frames);
    println!("  Range: [{:.3}, {:.3}]", 
        mel_data.iter().cloned().fold(f32::INFINITY, f32::min),
        mel_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    let device = if args.cpu { Device::Cpu } else { Device::cuda_if_available(0)? };
    println!("Device: {:?}", device);

    // Create mel tensor [1, 1, n_mels, n_frames]
    let mel = Tensor::from_slice(&mel_data, (n_mels, n_frames), &device)?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .to_dtype(DType::F32)?;
    println!("Mel tensor: {:?}", mel.dims());

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

    let mut before: Vec<u32> = vec![IM_START];
    before.extend(sys_enc.get_ids());
    before.push(IM_END);
    before.push(198);
    before.push(IM_START);
    before.extend(user_pre.get_ids());
    before.push(AUDIO_START);

    let mut after: Vec<u32> = vec![AUDIO_END];
    after.extend(user_post.get_ids());
    after.push(IM_END);
    after.push(198);
    after.push(IM_START);
    after.extend(asst.get_ids());

    println!("Tokens: {} + {} audio + {}", before.len(), audio_embeds.dim(1)?, after.len());

    let before_t = Tensor::from_slice(&before, (1, before.len()), &device)?;
    let after_t = Tensor::from_slice(&after, (1, after.len()), &device)?;
    let before_emb = thinker.embed_tokens(&before_t)?;
    let after_emb = thinker.embed_tokens(&after_t)?;

    let combined = Tensor::cat(&[&before_emb, &audio_embeds, &after_emb], 1)?;
    println!("Combined: {:?}", combined.dims());

    let output = thinker.forward_embeddings(&combined)?;
    let (_, seq_len, _) = output.text_logits.dims3()?;

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
