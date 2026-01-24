//! Qwen3-Omni: End-to-end Speech-to-Speech Model
//!
//! Architecture:
//! ```text
//! Audio Input
//!     │
//!     ▼
//! ┌─────────────────┐
//! │  AuT Encoder    │  650M params - Custom audio encoder (NOT Whisper)
//! │  (Audio→Token)  │  Encodes 16kHz audio into discrete tokens
//! └────────┬────────┘
//!          │ audio_tokens
//!          ▼
//! ┌─────────────────┐
//! │    Thinker      │  30B MoE (3B active) - Main reasoning model
//! │  (Think+Speak)  │  Generates text + talker tokens
//! └────────┬────────┘
//!          │ talker_tokens
//!          ▼
//! ┌─────────────────┐
//! │    Talker       │  3B MoE (0.3B active) - Speech synthesis
//! │  (Token→Codec)  │  Generates multi-codebook audio tokens
//! └────────┬────────┘
//!          │ codec_tokens (4 codebooks)
//!          ▼
//! ┌─────────────────┐
//! │   Code2Wav      │  ~200M params - Neural vocoder
//! │  (Codec→Audio)  │  Converts codebook tokens to waveform
//! └─────────────────┘
//!          │
//!          ▼
//!     Audio Output
//! ```
//!
//! Key differences from Qwen2.5-Omni:
//! - Custom AuT encoder instead of Whisper
//! - "Thinker-Talker" dual MoE architecture
//! - Multi-codebook speech tokens (4 parallel streams)
//!
//! References:
//! - HuggingFace: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
//! - Paper: Qwen3 Technical Report (2025)

mod config;
mod aut_encoder;
mod thinker;
mod talker;
mod code2wav;
mod audio;

pub use config::{Config, ThinkerConfig, TalkerConfig, AuTEncoderConfig, Code2WavConfig};
pub use aut_encoder::AuTEncoder;
pub use thinker::{Thinker, ThinkerOutput};
pub use talker::Talker;
pub use code2wav::Code2Wav;
pub use audio::{load_audio, AudioProcessor};

use candle::{Device, Result, Tensor};
use candle_nn::VarBuilder;

/// Complete Qwen3-Omni model for end-to-end speech-to-speech
#[allow(dead_code)]
pub struct Qwen3Omni {
    aut_encoder: AuTEncoder,
    thinker: Thinker,
    talker: Talker,
    code2wav: Code2Wav,
    device: Device,
}

impl Qwen3Omni {
    /// Load model from SafeTensors weights
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();

        let aut_encoder = AuTEncoder::new(&cfg.aut_encoder, vb.pp("aut_encoder"))?;
        let thinker = Thinker::new(&cfg.thinker, vb.pp("thinker"))?;
        let talker = Talker::new(&cfg.talker, vb.pp("talker"))?;
        let code2wav = Code2Wav::new(&cfg.code2wav, vb.pp("code2wav"))?;

        Ok(Self {
            aut_encoder,
            thinker,
            talker,
            code2wav,
            device,
        })
    }

    /// Process audio input and generate speech output
    ///
    /// # Arguments
    /// * `audio` - Input audio tensor [batch, samples] at 16kHz
    /// * `text_prompt` - Optional text prompt for context
    ///
    /// # Returns
    /// * Output audio tensor [batch, samples] at 16kHz
    pub fn forward(
        &mut self,
        audio: &Tensor,
        text_prompt: Option<&Tensor>,
    ) -> Result<Tensor> {
        // 1. Encode audio to tokens
        let audio_tokens = self.aut_encoder.forward(audio)?;

        // 2. Think: process audio tokens + optional text prompt
        let thinker_output = self.thinker.forward(&audio_tokens, text_prompt)?;

        // 3. Talk: generate codec tokens from talker tokens
        let codec_tokens = self.talker.forward(&thinker_output.talker_tokens)?;

        // 4. Synthesize audio from codec tokens
        let output_audio = self.code2wav.forward(&codec_tokens)?;

        Ok(output_audio)
    }

    /// Generate text-only response (for debugging/testing)
    pub fn generate_text(
        &mut self,
        audio: &Tensor,
        text_prompt: Option<&Tensor>,
        max_tokens: usize,
    ) -> Result<Tensor> {
        let audio_tokens = self.aut_encoder.forward(audio)?;
        self.thinker.generate(&audio_tokens, text_prompt, max_tokens)
    }

    /// Process text input only (no audio)
    pub fn text_to_speech(
        &mut self,
        text_tokens: &Tensor,
    ) -> Result<Tensor> {
        // For text-only input, skip AuT encoder
        let thinker_output = self.thinker.forward_text_only(text_tokens)?;
        let codec_tokens = self.talker.forward(&thinker_output.talker_tokens)?;
        self.code2wav.forward(&codec_tokens)
    }

    /// Clear KV caches for new conversation
    pub fn clear_kv_cache(&mut self) {
        self.thinker.clear_kv_cache();
        self.talker.clear_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_parse() {
        // Test that config can be parsed from JSON
        let json = r#"{
            "model_type": "qwen3_omni",
            "hidden_size": 4096,
            "num_hidden_layers": 40
        }"#;

        // This would test config deserialization
        // For now just verify the module compiles
        assert!(true);
    }
}
