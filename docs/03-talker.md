# Talker: 3B MoE Speech Synthesis Model

## Overview

Talker is a specialized MoE model for speech synthesis that receives talker tokens from Thinker and generates multi-codebook audio tokens for Code2Wav vocoder.

**Key Characteristics:**
- **Total Parameters**: 3B (Mixture of Experts)
- **Active Parameters**: 0.3B (10% - only selected experts)
- **Layers**: 24 transformer decoder layers
- **Architecture**: ALL layers use MoE (unlike Thinker which alternates)

## Configuration

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

## Structure

```rust
pub struct Talker {
    embed_tokens: candle_nn::Embedding,  // 4096 vocab -> 1536 hidden
    layers: Vec<DecoderLayer>,           // 24 layers (all MoE)
    norm: RmsNorm,
    codebook_head: CodebookHead,         // 4 parallel heads
    rotary: Arc<RotaryEmbedding>,
    device: Device,
    dtype: DType,
}
```

## Comparison: Talker vs Thinker

| Parameter | Thinker | Talker | Ratio |
|-----------|---------|--------|-------|
| Model size | 30B | 3B | 10x smaller |
| Active params | 3B (10%) | 0.3B (10%) | 10x smaller |
| Hidden size | 4096 | 1536 | -62.5% |
| Layers | 40 | 24 | -40% |
| Experts | 64 | 32 | -50% |
| MoE intermediate | 2816 | 1024 | -63.6% |
| GQA heads | 8/32 | 4/12 | same ratio |
| MoE distribution | Every 2nd layer | ALL layers | Different |

## Connection with Thinker

```
Thinker Output:
├─ text_logits: [batch, seq, vocab]     -> for text response
├─ talker_tokens: [batch, seq]          <- USED BY TALKER
└─ hidden_states: [batch, seq, hidden]

     |
     v (talker_tokens)

Talker Input:
  talker_tokens: [batch, seq]
    └─ Pass through embed_tokens (vocab 4096 -> hidden 1536)
```

In Thinker:
```rust
// Talker head: project to talker token space
let talker_head = linear_no_bias(cfg.hidden_size, 4096, vb.pp("talker_head"))?;

// After processing through all layers:
let talker_logits = self.talker_head.forward(&hidden)?;  // [batch, seq, 4096]
let talker_tokens = talker_logits.argmax(D::Minus1)?;    // [batch, seq]
```

## MoE Architecture (All Layers)

**Key difference from Thinker:** Talker uses MoE in **every** layer, not alternating.

```rust
struct SparseMoE {
    gate: Linear,                // [hidden_size -> num_experts]
    experts: Vec<MLPExpert>,     // 32 experts
    num_experts_per_tok: usize,  // 4 experts per token
}
```

### Routing Process

1. **Compute router logits**: `gate.forward(xs_flat)` -> [batch*seq, 32]
2. **Softmax weights**: normalize logits to probabilities
3. **Select top-k experts**: `arg_sort_last_dim(false)` -> top-4 experts
4. **Normalize weights**: divide by sum for selected experts
5. **Process through experts**: group tokens by expert, apply MLPs

## CodebookHead

CodebookHead generates discrete tokens for 4 parallel codebooks.

```rust
struct CodebookHead {
    heads: Vec<Linear>,          // 4 independent Linear layers
    num_codebooks: usize,        // 4
    codebook_size: usize,        // 2048
}
```

### Forward Pass

```rust
fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
    let mut codebook_tokens = Vec::with_capacity(4);

    for head in &self.heads {
        let logits = head.forward(hidden)?;      // [batch, seq, 2048]
        let tokens = logits.argmax(D::Minus1)?;  // [batch, seq]
        codebook_tokens.push(tokens);
    }

    Tensor::stack(&codebook_tokens, 2)           // [batch, seq, 4]
}
```

### Why 4 Codebooks?

Hierarchical audio encoding (similar to Encodec, HiFi-GAN):
- **Codebook 0**: Low-frequency (bass, voice tone)
- **Codebook 1**: Mid-frequency
- **Codebook 2**: High-frequency (consonants)
- **Codebook 3**: Fine details (artifacts, noise)

Each codebook: 2048 vocab (11 bits per token) x 4 = 44 bits per frame

## Forward Pass Flow

```
Input: talker_tokens [batch=2, seq=50]
       Values in range [0, 4096)
       These tokens come from Thinker.talker_head

     | EMBEDDING
     v

Embeddings [2, 50, 1536]
     |
     +-> embed_tokens: Linear(4096 -> 1536)

     | CAUSAL MASK
     v

Mask [1, 1, 50, 50]
     |
     +-> Lower triangular matrix for autoregression

     | LAYERS 0-23
     v

Hidden [2, 50, 1536]  <- after each layer
     |
     +-> DecoderLayer.forward():
         |- LayerNorm (ln1)
         |- Attention (self_attn)
         |  |- q_proj: [2, 50, 1536] -> [2, 12, 50, 128]
         |  |- k_proj: [2, 50, 1536] -> [2, 4, 50, 128]
         |  |- v_proj: [2, 50, 1536] -> [2, 4, 50, 128]
         |  |- Apply rotary embeddings
         |  |- GQA (repeat KV 12/4=3 times)
         |  +-> o_proj
         |- Residual: hidden + attn_out
         |- LayerNorm (ln2)
         |- SparseMoE.forward()
         |  |- gate: [2*50, 1536] -> [2*50, 32]
         |  |- Top-4 experts per token
         |  |- Normalize weights
         |  +-> Process through selected experts
         +-> Residual: hidden + moe_out

     | FINAL NORM
     v

Hidden [2, 50, 1536]

     | CODEBOOK HEAD
     v

Output: codec_tokens [2, 50, 4]
     |
     +-> Head 0: [2, 50, 1536] -> [2, 50, 2048] -> argmax -> [2, 50]
     +-> Head 1: ...
     +-> Head 2: ...
     +-> Head 3: ...
     +-> Stack -> [2, 50, 4]
```

## Architecture Summary

```
TALKER MODEL (3B MoE, 0.3B active)
|
+- Embedding Layer
|  +- 4096 vocab -> 1536 hidden
|
+- 24 DecoderLayers (ALL with MoE)
|  |- LayerNorm (pre-attention)
|  |- Grouped Query Attention (12 heads, 4 KV heads, head_dim=128)
|  |- Residual Connection
|  |- LayerNorm (pre-MoE)
|  |- Sparse MoE Block
|  |  |- Router (gate): 1536 -> 32 experts
|  |  |- Select top-4 experts per token
|  |  +- 32 MLP Experts (1024 hidden each)
|  +- Residual Connection
|
+- Final LayerNorm
|
+- CodebookHead (4 parallel)
   |- Head 0: 1536 -> 2048 -> argmax -> token [0, 2048)
   |- Head 1: 1536 -> 2048 -> argmax
   |- Head 2: 1536 -> 2048 -> argmax
   +- Head 3: 1536 -> 2048 -> argmax

OUTPUT: [batch, seq, 4] codec tokens
        | 320x upsampling
        | Code2Wav vocoder
        v
        16kHz audio waveform
```
