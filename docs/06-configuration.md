# Configuration Reference

## Config Hierarchy

```
Config (top-level)
├── AuTEncoderConfig
├── ThinkerConfig
├── TalkerConfig
└── Code2WavConfig
```

## Model Dimensions Summary

| Parameter | AuT Encoder | Thinker | Talker | Code2Wav |
|-----------|------------|---------|--------|----------|
| Hidden Size | 1024 | 4096 | 1536 | 512 |
| Num Layers | 24 | 40 | 24 | 4 (upsample) |
| Attention Heads | 16 | 32 | 12 | - |
| KV Heads (GQA) | - | 8 (4x) | 4 (3x) | - |
| Head Dimension | 64 | 128 | 128 | - |
| MoE Experts | - | 64 | 32 | - |
| Active per Token | - | 4/64 (6.25%) | 4/32 (12.5%) | - |

## AuTEncoderConfig

```rust
pub struct AuTEncoderConfig {
    pub hidden_size: usize,           // 1024
    pub num_hidden_layers: usize,     // 24
    pub num_attention_heads: usize,   // 16
    pub frame_size: usize,            // 400 (25ms @ 16kHz)
    pub hop_size: usize,              // 160 (10ms @ 16kHz)
    pub n_mels: usize,                // 128
    pub n_fft: usize,                 // 512
    pub audio_vocab_size: usize,      // 4096
    pub rms_norm_eps: f64,            // 1e-6
}
```

## ThinkerConfig

```rust
pub struct ThinkerConfig {
    pub hidden_size: usize,              // 4096
    pub num_hidden_layers: usize,        // 40
    pub num_attention_heads: usize,      // 32
    pub num_key_value_heads: usize,      // 8 (GQA)
    pub head_dim: usize,                 // 128
    pub intermediate_size: usize,        // 11008 (dense MLP)
    pub moe_intermediate_size: usize,    // 2816 (MoE experts)
    pub num_experts: usize,              // 64
    pub num_experts_per_tok: usize,      // 4
    pub decoder_sparse_step: usize,      // 2 (MoE every 2nd layer)
    pub norm_topk_prob: bool,            // true
    pub max_position_embeddings: usize,  // 32768
    pub rope_theta: f64,                 // 1_000_000.0
    pub rms_norm_eps: f64,               // 1e-6
    pub vocab_size: usize,               // 151936
    pub tie_word_embeddings: bool,       // false
    pub hidden_act: Activation,          // SiLU
}
```

## TalkerConfig

```rust
pub struct TalkerConfig {
    pub hidden_size: usize,              // 1536
    pub num_hidden_layers: usize,        // 24
    pub num_attention_heads: usize,      // 12
    pub num_key_value_heads: usize,      // 4 (GQA)
    pub head_dim: usize,                 // 128
    pub moe_intermediate_size: usize,    // 1024
    pub num_experts: usize,              // 32
    pub num_experts_per_tok: usize,      // 4
    pub num_codebooks: usize,            // 4
    pub codebook_size: usize,            // 2048
    pub rms_norm_eps: f64,               // 1e-6
    pub hidden_act: Activation,          // SiLU
}
```

## Code2WavConfig

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

## Audio Parameters

```
sample_rate: 16000 Hz
frame_size: 400 samples (25ms)
hop_size: 160 samples (10ms)
codec_rate: 50 Hz (320x upsampling in Code2Wav)
```

## Special Tokens

| Token Space | Vocab Size | Used By |
|-------------|-----------|---------|
| Text tokens | 151936 | Thinker (input/output) |
| Audio tokens | 4096 | AuT Encoder → Thinker |
| Talker tokens | 4096 | Thinker → Talker |
| Codebook tokens | 2048 × 4 | Talker → Code2Wav |

## Synchronization Points

These values **MUST match** between configs:

| Parameter | Talker | Code2Wav |
|-----------|--------|----------|
| `num_codebooks` | 4 | 4 |
| `codebook_size` | 2048 | 2048 |
