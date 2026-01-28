# Module Structure

## File Overview

| File | Size | Purpose |
|------|------|---------|
| `mod.rs` | 5.2 KB | Main module, Qwen3Omni struct, component integration |
| `config.rs` | 11.3 KB | Configuration structures for all components |
| `thinker.rs` | 18.7 KB | 30B MoE model (3B active) - main reasoning |
| `talker.rs` | 13.8 KB | 3B MoE model (0.3B active) - speech synthesis |
| `aut_encoder.rs` | 11.6 KB | Custom audio-to-token encoder (650M params) |
| `code2wav.rs` | 7.1 KB | Neural vocoder (~200M params) |
| `audio.rs` | 7.6 KB | Audio utilities (mel spectrogram, I/O, resampling) |

**Total:** ~75 KB Rust code, 7 files

## Dependency Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Audio Input (16kHz)                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ load_audio + AudioProcessor
                         │ pcm_to_mel (mel spectrogram)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  AuT Encoder                                 │
│  (Audio-to-Token Encoder) - 650M params                     │
│  ────────────────────────────────────────────                │
│  - ConvStem: 2 conv1d layers (downsampling)                 │
│  - 24 EncoderLayer (Attention + MLP)                        │
│  - VectorQuantizer: discrete tokens [batch, seq]           │
│  - Config: hidden=1024, vocab=4096, n_mels=128             │
└────────────────────────┬─────────────────────────────────────┘
                         │ audio_tokens [batch, audio_seq]
                         │
        ┌────────────────┴────────────────┐
        │                                 │
        ▼ (+ optional text_prompt)        │
┌──────────────────────────────────────────────────────────────┐
│                    Thinker (Brain)                           │
│  30B MoE (3B active) - Main reasoning model                 │
│  ────────────────────────────────────────                   │
│  - Token Embeddings: vocab_size=151936                      │
│  - Audio Embedding: linear projection                       │
│  - 40 DecoderLayers (alternating attention + MLP/MoE)      │
│  - MoE: 64 experts, 4 active per token, sparse_step=2      │
│  - Outputs: text_logits + talker_tokens                    │
│  - Config: hidden=4096, heads=32, kv_heads=8              │
└────────────────────────┬─────────────────────────────────────┘
                         │ talker_tokens [batch, seq]
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   Talker (Voice)                             │
│  3B MoE (0.3B active) - Speech synthesis                    │
│  ────────────────────────────────────────                   │
│  - Embedding: talker_vocab=4096                             │
│  - 24 DecoderLayers (all with MoE)                         │
│  - MoE: 32 experts, 4 active per token                      │
│  - CodebookHead: 4 parallel output heads                    │
│  - Outputs: codec_tokens [batch, seq, 4 codebooks]        │
│  - Config: hidden=1536, heads=12, kv_heads=4              │
└────────────────────────┬─────────────────────────────────────┘
                         │ codec_tokens [batch, seq, 4]
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Code2Wav (Vocoder)                          │
│  Neural vocoder - ~200M params                              │
│  ────────────────────────────────────────                   │
│  - CodebookEmbedding: embeds all 4 codebooks              │
│  - 4 UpsampleBlock: rates [8, 5, 4, 2] (total 320x)       │
│  - Each block: TransposedConv1d + 3 ResidualBlock         │
│  - Output conv: to mono audio                              │
│  - Config: hidden=512, codebook_size=2048                 │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│               Audio Output (16kHz mono)                      │
└──────────────────────────────────────────────────────────────┘
```

## Public API

### mod.rs (Main Interface)

```rust
pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self>
    // Load complete Qwen3Omni from SafeTensors

pub fn forward(&mut self, audio: &Tensor, text_prompt: Option<&Tensor>)
    -> Result<Tensor>
    // Speech-to-Speech: audio [batch, samples] -> audio [batch, samples]

pub fn generate_text(&mut self, audio: &Tensor, text_prompt: Option<&Tensor>,
                     max_tokens: usize) -> Result<Tensor>
    // Speech-to-Text: generate text tokens only (for debugging)

pub fn text_to_speech(&mut self, text_tokens: &Tensor) -> Result<Tensor>
    // Text-to-Speech: skip AuT encoder, use only Talker+Code2Wav

pub fn clear_kv_cache(&mut self)
    // Clear KV caches for new session
```

### aut_encoder.rs

```rust
pub fn new(cfg: &AuTEncoderConfig, vb: VarBuilder) -> Result<Self>

pub fn forward(&self, mel: &Tensor) -> Result<Tensor>
    // mel [batch, n_mels, frames] -> tokens [batch, seq_len]

pub fn encode_continuous(&self, mel: &Tensor) -> Result<Tensor>
    // mel -> embeddings [batch, seq, hidden] (before quantization)
```

### thinker.rs

```rust
pub fn new(cfg: &ThinkerConfig, vb: VarBuilder) -> Result<Self>

pub fn forward(&mut self, audio_tokens: &Tensor, text_prompt: Option<&Tensor>)
    -> Result<ThinkerOutput>

pub fn forward_text_only(&mut self, text_tokens: &Tensor) -> Result<ThinkerOutput>

pub fn generate(&mut self, audio_tokens: &Tensor, text_prompt: Option<&Tensor>,
                max_tokens: usize) -> Result<Tensor>

pub fn clear_kv_cache(&mut self)
```

### talker.rs

```rust
pub fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self>

pub fn forward(&mut self, talker_tokens: &Tensor) -> Result<Tensor>
    // talker_tokens [batch, seq] -> codec_tokens [batch, seq, 4]

pub fn clear_kv_cache(&mut self)
```

### code2wav.rs

```rust
pub fn new(cfg: &Code2WavConfig, vb: VarBuilder) -> Result<Self>

pub fn forward(&self, codec_tokens: &Tensor) -> Result<Tensor>
    // codec_tokens [batch, seq, 4] -> audio [batch, samples]
    // Upsample by 320x: 50Hz codec rate -> 16kHz
```

### audio.rs

```rust
pub fn new(sample_rate: usize, n_fft: usize, hop_length: usize,
           n_mels: usize, device: &Device) -> Self

pub fn pcm_to_mel(&self, pcm: &Tensor) -> Result<Tensor>
    // raw audio -> mel spectrogram [batch, n_mels, frames]

pub fn load_audio(data: &[u8], target_sample_rate: usize) -> Result<Vec<f32>>
    // WAV file -> [f32] at 16kHz
```

## Dependencies

```rust
// Main dependencies:
use candle::{Device, Tensor, DType, Result}
use candle_nn::{VarBuilder, Module, Embedding, Conv1d, ...}

// Internal modules:
use crate::models::with_tracing::{linear_no_bias, Linear, RmsNorm}
use crate::models::whisper::audio::log_mel_spectrogram_

// Standard library:
use std::sync::Arc
use serde::Deserialize
```
