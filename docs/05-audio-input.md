# Audio Input Pipeline (ASR Direction)

## Overview

The audio input pipeline converts raw WAV audio to discrete tokens for the Thinker model.

```
WAV File (16kHz)
    |
    v
[load_audio] -> Vec<f32>
    |
    v
[pcm_to_mel] -> Mel Spectrogram [batch, 128, frames]
    |
    v
[AuTEncoder] -> Audio Tokens [batch, seq] in [0, 4095]
    |
    v
[Thinker] -> Text + Talker tokens
```

## WAV Loading

### Function: `load_audio`

```rust
pub fn load_audio(data: &[u8], target_sample_rate: usize) -> Result<Vec<f32>>
```

**Process:**

1. **Format validation**:
   - Check RIFF header (4 bytes: `b"RIFF"`)
   - Check WAVE identifier (position 8-12: `b"WAVE"`)

2. **Parse chunks**:
   - Extract from fmt chunk:
     - `num_channels`: mono/stereo
     - `sample_rate`: original sample rate
     - `bits_per_sample`: only 16-bit supported

3. **PCM to Float conversion**:
   ```rust
   sample = (bytes[i] as i16) as f32 / 32768.0  // Range [-1, 1]
   ```

4. **Stereo to Mono**:
   ```rust
   if num_channels == 2:
       mono[i] = (left[i] + right[i]) / 2.0
   ```

5. **Resampling** (linear interpolation):
   ```rust
   ratio = from_rate / to_rate
   new_sample[i] = lerp(old_samples, i * ratio)
   ```

## Mel Spectrogram

### AudioProcessor

```rust
pub struct AudioProcessor {
    sample_rate: usize,   // 16000
    n_fft: usize,         // 512
    hop_length: usize,    // 160 (10ms stride)
    n_mels: usize,        // 128
    device: Device,
}
```

### Parameters

```
sample_rate: 16000         // 16 kHz
n_fft: 512                 // 25ms window @ 16kHz
n_mels: 128                // Number of mel filters
hop_length: 160            // 10ms stride
frame_size: 400            // 25ms window (samples)
```

### Output Size

For 1 second audio @ 16kHz:
- 16000 samples -> n_frames = 16000/160 = 100
- Output: `[1, 128, 100]`

### Mel Filterbank

```
1. Hz to Mel conversion:
   hz_to_mel(hz) = 2595 * log10(1 + hz/700)

2. Create n_mels + 2 equally spaced points on mel scale

3. Convert to FFT bins:
   bin = (n_fft + 1) * hz / sample_rate

4. Create triangular filters with 50% overlap
```

Output shape: `[n_mels, 1+n_fft/2]` = `[128, 257]`

## AuT Encoder (Audio-to-Token)

### Architecture

```
Mel Spectrogram [batch, 128, frames]
    |
    v
+---------------------+
|   Conv Stem         |  2 conv layers
| - Conv1d(128->512)  |  kernel=7, stride=2
| - Conv1d(512->1024) |  kernel=3, stride=2
+----------+----------+
           | [batch, 1024, frames/4]
           v
    +------+------+
    | Transformer |  24 encoder layers
    |   Layers    |
    +------+------+
           | [batch, frames/4, 1024]
           v
    +------+------+
    |   RmsNorm   |
    +------+------+
           | [batch, frames/4, 1024]
           v
    +------+------+
    |  Quantizer  |  VQ-VAE style
    +------+------+
           |
           v
   Audio tokens [batch, frames/4]
```

### Configuration

```rust
pub struct AuTEncoderConfig {
    pub hidden_size: usize,           // 1024
    pub num_hidden_layers: usize,     // 24
    pub num_attention_heads: usize,   // 16
    pub frame_size: usize,            // 400
    pub hop_size: usize,              // 160
    pub n_mels: usize,                // 128
    pub n_fft: usize,                 // 512
    pub audio_vocab_size: usize,      // 4096
    pub rms_norm_eps: f64,            // 1e-6
}
```

### ConvStem

```rust
struct ConvStem {
    conv1: Conv1d,   // 128 -> 512, kernel=7, stride=2, padding=3
    conv2: Conv1d,   // 512 -> 1024, kernel=3, stride=2, padding=1
    norm: LayerNorm,
}
```

**Downsampling**: 2 x 2 = 4x total
- Mel frames: 100 -> 25 tokens
- Temporal resolution: 10ms/frame x 4 = 40ms/token

### Transformer Encoder Layer

```rust
struct EncoderLayer {
    attn: Attention,     // Self-attention with RoPE
    mlp: MLP,            // SwiGLU MLP
    ln1: RmsNorm,
    ln2: RmsNorm,
}
```

Each layer:
1. Pre-norm RmsNorm
2. Multi-Head Attention (16 heads, head_dim=64)
3. RoPE positional embeddings
4. MLP: gate_proj(1024->4096) * SiLU + up_proj -> down_proj

### Vector Quantizer

```rust
struct VectorQuantizer {
    embedding: Embedding,  // [4096, 1024]
    num_embeddings: 4096,
}
```

**Quantization Process:**

```
1. Compute L2 distances:
   distance = ||x - e||^2 = ||x||^2 - 2*x*e + ||e||^2

2. Find nearest embedding:
   indices = argmin(distances)  -> [batch, seq] in [0, 4095]

3. Look up quantized embeddings:
   quantized = embedding(indices)  -> [batch, seq, 1024]
```

### Forward Pass

```rust
pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
    // mel: [batch, n_mels=128, frames]

    // 1. Conv stem (4x downsampling)
    let xs = self.conv_stem.forward(mel)?;
    // -> [batch, frames/4, 1024] after transpose

    // 2. Transformer layers
    for layer in &self.layers {
        xs = layer.forward(&xs, &self.rotary, 0)?;
    }

    // 3. Final norm
    xs = self.norm.forward(&xs)?;

    // 4. Quantize to discrete tokens
    let (_, tokens) = self.quantizer.forward(&xs)?;
    // -> [batch, frames/4] in [0, 4095]

    Ok(tokens)
}
```

## Integration with Thinker

### Audio Embedding Path

```rust
// In Thinker
embed_tokens: Embedding(vocab_size=151936, hidden_size=4096)
audio_embed: Linear(hidden_size -> hidden_size)
```

### Thinker Forward with Audio

```rust
pub fn forward(&mut self, audio_tokens: &Tensor, text_prompt: Option<&Tensor>) {
    // 1. Embed audio tokens (shared embedding with text)
    let audio_embeds = self.embed_tokens.forward(audio_tokens)?;
    // -> [batch, seq_len, 4096]

    // 2. Project through audio_embed layer
    let audio_embeds = self.audio_embed.forward(&audio_embeds)?;
    // -> [batch, seq_len, 4096]

    // 3. Combine with optional text prompt
    let embeddings = match text_prompt {
        Some(text) => {
            let text_embeds = self.embed_tokens.forward(text)?;
            Tensor::cat(&[audio_embeds, text_embeds], 1)?
        }
        None => audio_embeds
    };

    // 4. Process through 40 transformer layers...
}
```

## NOT Whisper-Based

AuT Encoder is a **custom architecture**, not based on Whisper.

### Differences from Whisper

| Aspect | AuT | Whisper |
|--------|-----|---------|
| Design | Custom for Qwen3 | General ASR |
| Size | 650M (24 layers) | 390M (12 layers) |
| Mel filters | 128 | 80 |
| FFT | 512 | 400 |
| Frame rate | 100 fps (10ms) | 80 fps (12.5ms) |
| Output | Discrete tokens (VQ-VAE) | BPE tokens |
| Vocab size | 4096 | 50258 |
| Positional | RoPE | Sinusoidal |

## Complete Data Flow

```
Input WAV (16kHz)
    |
    v
+-------------------+
| load_audio()      |
| - Parse WAV       |
| - Convert PCM     |
| - Stereo->Mono    |
| - Resample        |
+--------+----------+
         | Vec<f32>
         v
+-------------------+
| AudioProcessor    |
| pcm_to_mel()      |
| - Mel filterbank  |
| - STFT via FFT    |
| - Log scale       |
+--------+----------+
         | [batch, 128, frames]
         v
+-------------------+
| AuTEncoder        |
| - Conv stem (4x)  |
| - 24 transformer  |
| - VQ quantizer    |
+--------+----------+
         | [batch, frames/4]
         | dtype: u32 in [0, 4095]
         v
+-------------------+
| Thinker           |
| - embed_tokens    |
| - audio_embed     |
| - 40 layers (MoE) |
+--------+----------+
         | [batch, seq, 4096]
         v
+-------------------+
| Output Heads      |
| - text_logits     |
| - talker_tokens   |
+-------------------+
```
