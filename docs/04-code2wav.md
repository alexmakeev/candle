# Code2Wav: Neural Vocoder

## Overview

Code2Wav is a HiFi-GAN-like neural vocoder that converts discrete codec tokens from Talker into audio waveforms.

**Key Characteristics:**
- **Parameters**: ~5.7M (lightweight vocoder)
- **Upsampling**: 320x total (8 x 5 x 4 x 2)
- **Input**: 4-codebook tokens at 50 Hz
- **Output**: 16 kHz mono audio

## Configuration

```rust
pub struct Code2WavConfig {
    pub hidden_size: usize,           // 512
    pub num_upsample_layers: usize,   // 4
    pub upsample_rates: Vec<usize>,   // [8, 5, 4, 2] = 320x total
    pub num_residual_blocks: usize,   // 3
    pub num_codebooks: usize,         // 4 (must match Talker)
    pub codebook_size: usize,         // 2048 (must match Talker)
}
```

## Architecture

```
Codec Tokens [batch, seq, 4 codebooks]
    |
    v
CodebookEmbedding (distributed encoding)
    |
    v
[batch, hidden=512, seq]
    |
    v
UpsampleBlock x 4 (progressive upsampling)
    |
    v
Output Conv (project to 1 channel)
    |
    v
Tanh (normalization)
    |
    v
Waveform [batch, samples]
```

## CodebookEmbedding

Converts 4 independent codebook token streams into unified hidden representation.

```rust
struct CodebookEmbedding {
    embeddings: Vec<candle_nn::Embedding>,  // 4 embeddings
    proj: candle_nn::Conv1d,                // 512 -> 512, kernel=1
}
```

### Process

```
Input: [batch, seq, num_codebooks=4]

1. Distributed embedding:
   For each codebook i:
   - Extract: tokens[i] -> [batch, seq]
   - Embed: [batch, seq] -> [batch, seq, 128]  (512/4 = 128 dims each)

2. Concatenate along hidden dimension:
   [batch, seq, 128] x 4 -> [batch, seq, 512]

3. Transpose for convolution:
   [batch, seq, 512] -> [batch, 512, seq]

4. Projection (Conv1d 1x1):
   [batch, 512, seq] -> [batch, 512, seq]
```

## UpsampleBlocks

Progressive upsampling with residual refinement.

### Configuration

| Layer | In Channels | Out Channels | Upsample Rate | Output Seq |
|-------|-------------|--------------|---------------|------------|
| 0 | 512 | 256 | 8 | seq x 8 |
| 1 | 256 | 128 | 5 | seq x 40 |
| 2 | 128 | 64 | 4 | seq x 160 |
| 3 | 64 | 32 | 2 | seq x 320 |

### UpsampleBlock Structure

```
Input: [batch, channels_in, seq]
    |
    v
[1] ConvTranspose1d (deconvolution)
    - stride = upsample_rate
    - kernel_size = 2 x upsample_rate
    - padding = upsample_rate / 2
    |
    v
[batch, channels_out, seq x upsample_rate]
    |
    v
[2] Residual Blocks (3 per layer)
    - Dilated convolutions (1, 3, 9)
    |
    v
Output: [batch, channels_out, seq x upsample_rate]
```

### ResidualBlock

```
Input: [batch, channels, seq]
    |
    v
GELU
    |
    v
Conv1d(kernel=3, dilation=d, padding=d)
    |
    v
GELU
    |
    v
Conv1d(kernel=1, padding=0)
    |
    + residual connection
    |
    v
Output: [batch, channels, seq]
```

**Dilations**: 3^i where i in [0, 1, 2]
- Block 0: dilation = 1
- Block 1: dilation = 3
- Block 2: dilation = 9

Receptive field: 1 + 2x1 + 2x3 + 2x9 = 27 per upsample block

## Output Layer

```rust
let output_conv = candle_nn::conv1d(
    32,     // input channels (after layer 3)
    1,      // mono audio
    7,      // large kernel for smoothing
    Conv1dConfig { padding: 3, .. },
)?;
```

Final activation chain:
1. GELU on hidden
2. Conv1d to 1 channel
3. Tanh for [-1, 1] normalization
4. Squeeze channel dimension

## Forward Pass

```rust
pub fn forward(&self, codec_tokens: &Tensor) -> Result<Tensor> {
    // Embed codec tokens: [batch, hidden, seq]
    let mut xs = self.codebook_embed.forward(codec_tokens)?;

    // Upsample through blocks
    for block in &self.upsample_blocks {
        xs = block.forward(&xs)?;
    }

    // Final activation and output projection
    let xs = xs.gelu_erf()?;
    let xs = self.output_conv.forward(&xs)?;

    // Apply tanh for audio normalization
    let xs = xs.tanh()?;

    // Squeeze channel dim: [batch, 1, samples] -> [batch, samples]
    xs.squeeze(1)
}
```

## Tensor Flow Example

For input [batch=1, seq=50, codebooks=4]:

```
Input:                       [1, 50, 4]
| CodebookEmbedding
Embedded:                    [1, 512, 50]

| UpsampleBlock 0 (8x)
After ConvTranspose1d:       [1, 256, 400]
After ResidualBlocks:        [1, 256, 400]

| UpsampleBlock 1 (5x)
After ConvTranspose1d:       [1, 128, 2000]
After ResidualBlocks:        [1, 128, 2000]

| UpsampleBlock 2 (4x)
After ConvTranspose1d:       [1, 64, 8000]
After ResidualBlocks:        [1, 64, 8000]

| UpsampleBlock 3 (2x)
After ConvTranspose1d:       [1, 32, 16000]
After ResidualBlocks:        [1, 32, 16000]

| Final Operations
After GELU:                  [1, 32, 16000]
After Output Conv:           [1, 1, 16000]
After Tanh:                  [1, 1, 16000] (values in [-1, 1])

| Squeeze(1)
Output Waveform:             [1, 16000]
```

## Audio Parameters

```
Codec frequency:     50 Hz
Sample rate:         16 kHz
Upsampling factor:   320x (8 x 5 x 4 x 2)

For seq_len = 50:
  Duration = 50 / 50 Hz = 1.0 sec
  Output samples = 50 x 320 = 16000 = 1.0 sec @ 16 kHz
```

## Parameter Count

| Component | Parameters |
|-----------|------------|
| CodebookEmbedding | 4 x (2048 x 128) + (512 x 512 x 1) = 1.3M |
| UpsampleBlocks[0] | Conv: 512x256x16, ResBlocks = 3.3M |
| UpsampleBlocks[1] | Conv: 256x128x10, ResBlocks = 828K |
| UpsampleBlocks[2] | Conv: 128x64x8, ResBlocks = 207K |
| UpsampleBlocks[3] | Conv: 64x32x4, ResBlocks = 52K |
| OutputConv | 32 x 1 x 7 = 224 |
| **Total** | **~5.7M** |

## Activation Functions

- **GELU**: in residual blocks and before output
- **Tanh**: only at output for audio normalization

No SnakeBeta activation in current implementation (simpler than HiFi-GAN).
