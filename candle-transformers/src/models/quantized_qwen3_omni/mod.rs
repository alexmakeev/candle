//! Quantized Qwen3-Omni model for GGUF inference
//!
//! This module provides quantized versions of Qwen3-Omni components
//! that load weights from GGUF files.
//!
//! Architecture:
//! - AudioTower: Encodes audio features (32 transformer layers)
//! - Thinker: Main reasoning MoE model (48 layers, 128 experts)
//! - Talker: Speech synthesis MoE (20 layers, 128 experts)
//! - Code2Wav: Neural vocoder for audio synthesis
//!
//! GGUF tensor naming:
//! - thinker.audio_tower.* - Audio encoder
//! - thinker.model.* - Main reasoning model
//! - thinker.lm_head.* - Text output head
//! - talker.* - Speech synthesis
//! - code2wav.* - Vocoder

mod gguf_loader;
mod config;
mod audio_tower;
mod thinker;
mod talker;
mod code2wav;

pub use gguf_loader::Gguf;
pub use config::{Config, Code2WavConfig, TalkerConfig, ThinkerConfig, AudioTowerConfig};
pub use audio_tower::AudioTower;
pub use thinker::{Thinker, ThinkerOutput};
pub use talker::{Talker, Speaker, TalkerSpecialTokens};
pub use code2wav::Code2Wav;

use candle::{Device, Result, Tensor};

/// Complete Qwen3-Omni model for GGUF inference
pub struct Qwen3Omni {
    pub audio_tower: AudioTower,
    pub thinker: Thinker,
    pub talker: Talker,
    pub code2wav: Code2Wav,
    #[allow(dead_code)]
    device: Device,
}

impl Qwen3Omni {
    /// Load model from GGUF file
    pub fn from_gguf<R: std::io::Read + std::io::Seek>(
        gguf: &mut Gguf<R>,
        device: &Device,
    ) -> Result<Self> {
        let cfg = gguf.config()?;

        let audio_tower = AudioTower::from_gguf(gguf, &cfg.audio_tower, device)?;
        let thinker = Thinker::from_gguf(gguf, &cfg.thinker, device)?;
        let talker = Talker::from_gguf(gguf, &cfg.talker, device)?;
        let code2wav = Code2Wav::from_gguf(gguf, &cfg.code2wav, device)?;

        Ok(Self {
            audio_tower,
            thinker,
            talker,
            code2wav,
            device: device.clone(),
        })
    }

    /// Process audio input and generate speech output
    pub fn forward(
        &mut self,
        audio_features: &Tensor,
        text_prompt: Option<&Tensor>,
    ) -> Result<Tensor> {
        // 1. Encode audio features
        let audio_embeds = self.audio_tower.forward(audio_features)?;

        // 2. Think: process embeddings
        let thinker_output = self.thinker.forward(&audio_embeds, text_prompt)?;

        // 3. Talk: generate codec tokens
        let codec_tokens = self.talker.forward(&thinker_output.hidden_states)?;

        // 4. Synthesize audio
        self.code2wav.forward(&codec_tokens)
    }

    /// Text-to-speech: convert text to audio
    pub fn text_to_speech(&mut self, text_tokens: &Tensor) -> Result<Tensor> {
        let thinker_output = self.thinker.forward_text_only(text_tokens)?;
        let codec_tokens = self.talker.forward(&thinker_output.hidden_states)?;
        self.code2wav.forward(&codec_tokens)
    }

    /// Clear KV caches
    pub fn clear_kv_cache(&mut self) {
        self.thinker.clear_kv_cache();
        self.talker.clear_kv_cache();
    }
}
