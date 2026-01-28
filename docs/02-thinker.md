# Thinker: 30B MoE Reasoning Model

## Overview

Thinker is the main "brain" of Qwen3-Omni, responsible for processing audio tokens and optional text prompts to generate text logits and talker tokens for speech synthesis.

**Key Characteristics:**
- **Total Parameters**: 30B (Mixture of Experts)
- **Active Parameters**: 3B (only selected experts process each token)
- **Layers**: 40 transformer decoder layers
- **Architecture**: GQA + alternating Dense MLP / Sparse MoE

## Configuration

```rust
pub struct ThinkerConfig {
    // Core dimensions
    pub hidden_size: usize,              // 4096
    pub num_hidden_layers: usize,        // 40
    pub num_attention_heads: usize,      // 32
    pub num_key_value_heads: usize,      // 8 (GQA)
    pub head_dim: usize,                 // 128

    // Dense MLP
    pub intermediate_size: usize,        // 11008

    // MoE configuration
    pub moe_intermediate_size: usize,    // 2816
    pub num_experts: usize,              // 64
    pub num_experts_per_tok: usize,      // 4
    pub decoder_sparse_step: usize,      // 2 (MoE every 2nd layer)
    pub norm_topk_prob: bool,            // true

    // Positional embeddings
    pub max_position_embeddings: usize,  // 32768
    pub rope_theta: f64,                 // 1_000_000.0

    // Normalization
    pub rms_norm_eps: f64,               // 1e-6
    pub hidden_act: Activation,          // SiLU

    // Vocabulary
    pub vocab_size: usize,               // 151936
    pub tie_word_embeddings: bool,       // false
}
```

## Structure

```rust
pub struct Thinker {
    embed_tokens: candle_nn::Embedding,  // [vocab_size, hidden_size]
    audio_embed: Linear,                 // hidden_size -> hidden_size
    layers: Vec<DecoderLayer>,           // 40 layers
    norm: RmsNorm,
    lm_head: Linear,                     // hidden_size -> vocab_size
    talker_head: Linear,                 // hidden_size -> 4096
    rotary: Arc<RotaryEmbedding>,
    device: Device,
    dtype: DType,
}

pub struct ThinkerOutput {
    pub text_logits: Tensor,         // [batch, seq, 151936]
    pub talker_tokens: Tensor,       // [batch, seq]
    pub hidden_states: Tensor,       // [batch, seq, 4096]
}
```

## Layer Distribution

```
Layer 0: Dense MLP
Layer 1: Sparse MoE  ← (0+1) % 2 == 0
Layer 2: Dense MLP
Layer 3: Sparse MoE
Layer 4: Dense MLP
Layer 5: Sparse MoE
...
Layer 38: Dense MLP
Layer 39: Sparse MoE

MoE layers: 1, 3, 5, 7, ... 39 (20 out of 40 layers)
```

## Grouped Query Attention (GQA)

GQA reduces KV cache by 75% while maintaining quality.

```
Input: [batch, seq_len, hidden_size=4096]

1. Linear projections:
   - Query: 4096 -> 32 heads x 128 dim = [batch, seq, 32, 128]
   - Key:   4096 -> 8 heads x 128 dim  = [batch, seq, 8, 128]   <- 4x compression!
   - Value: 4096 -> 8 heads x 128 dim  = [batch, seq, 8, 128]

2. Apply RoPE to Q and K

3. KV Cache: concatenate new K, V with previous cache

4. Repeat KV heads for attention:
   - K, V: [batch, 8, seq, 128] -> [batch, 32, seq, 128]

5. Scaled dot-product attention:
   scores = (Q @ K.T) / sqrt(128)
   attention = softmax(scores + causal_mask)
   output = attention @ V

6. Output projection: [batch, 32, seq, 128] -> [batch, seq, 4096]
```

## Sparse MoE Architecture

```rust
pub struct SparseMoE {
    gate: Linear,                // hidden_size -> num_experts (4096 -> 64)
    experts: Vec<MLPExpert>,     // 64 independent MLP experts
    num_experts_per_tok: usize,  // 4
    norm_topk_prob: bool,        // true
}

pub struct MLPExpert {
    gate_proj: Linear,           // 4096 -> 2816
    up_proj: Linear,             // 4096 -> 2816
    down_proj: Linear,           // 2816 -> 4096
    act_fn: Activation,          // SiLU
}
```

### MoE Forward Pass

```
Input: [batch, seq_len, hidden_size]

1. Reshape to 2D: [batch*seq_len, hidden_size]

2. Router logits:
   router_logits = gate(x)      -> [batch*seq, 64]
   routing_weights = softmax()  -> [batch*seq, 64]

3. Select top-k experts:
   top_indices = argsort(routing_weights)[:, :4]
   top_weights = gather(routing_weights, top_indices)

4. Normalize (optional):
   if norm_topk_prob:
       top_weights = top_weights / sum(top_weights)

5. Process through experts:
   result = zeros_like(x)
   for expert_idx in 0..64:
       if tokens_for_this_expert:
           expert_output = experts[expert_idx](selected_tokens)
           result += expert_output * weights

6. Reshape: [batch, seq_len, hidden_size]
```

## Forward Pass Flow

### Audio + Optional Text

```
1. Embed audio tokens:
   audio_embeds = embed_tokens(audio_tokens)
   audio_embeds = audio_embed(audio_embeds)   # Linear projection
   -> [batch, audio_seq_len, 4096]

2. Combine with text (if present):
   if text_prompt:
       text_embeds = embed_tokens(text_prompt)
       embeddings = cat([audio_embeds, text_embeds], dim=1)
   else:
       embeddings = audio_embeds

3. Create causal attention mask

4. Process through 40 layers:
   for layer in layers:
       hidden = layer.forward(hidden, rotary, mask, offset)

5. Final normalization:
   hidden = norm(hidden)

6. Generate outputs:
   text_logits = lm_head(hidden)       -> [batch, seq, 151936]
   talker_logits = talker_head(hidden) -> [batch, seq, 4096]
   talker_tokens = argmax(talker_logits)
```

### Autoregressive Generation

```rust
pub fn generate(&mut self, audio_tokens, text_prompt, max_tokens) -> Result<Tensor> {
    // Initial forward pass (full sequence)
    let output = self.forward(audio_tokens, text_prompt)?;
    let mut next_token = output.text_logits[-1].argmax();

    // Autoregressive loop
    for step in 1..max_tokens {
        let embeddings = embed_tokens(next_token);

        // Forward through all layers with offset (for RoPE)
        for layer in layers:
            hidden = layer.forward(hidden, rotary, None, offset);
            // KV cache is used and updated!

        hidden = norm(hidden);
        logits = lm_head(hidden);
        next_token = argmax(logits);

        // Check for EOS token
        if next_token == 2:
            break;

        offset += 1;
    }

    return concatenated_tokens;
}
```

## RoPE (Rotary Position Embeddings)

```rust
struct RotaryEmbedding {
    cos: Tensor,  // [max_seq, head_dim]
    sin: Tensor,
    dim: usize,   // head_dim = 128
}
```

Initialization:
```
theta_i = theta^(-2i/d) where theta=1M, d=128
freqs = positions @ inv_freq  -> [max_seq, 64]
cos, sin = cos(freqs), sin(freqs)  -> [max_seq, 128]
```

## Memory Requirements

```
Model weights:
- Embeddings: 622M params
- Dense layers: 4.3B params
- MoE layers: 24B params
- Output heads: 16.7M params
Total: ~29B params x 2 bytes (BF16) = 58 GB

KV Cache (with GQA, batch=1, seq=8192):
- Per layer: 8192 x 8 x 128 x 2 x 2 bytes = 4.2 MB
- Total 40 layers: 168 MB

Activation memory (peak): ~500 MB

Total for inference: 58 GB (weights) + 0.5 GB (activations + cache)
```

## Comparison with Other Architectures

| Aspect | Thinker | Qwen2.5 | LLaMA 3 |
|--------|---------|---------|---------|
| Architecture | Dense + Sparse MoE | Dense | Dense |
| Parameters | 30B (3B active) | 32B | 70B |
| Attention | GQA (32->8 heads) | GQA (32->8) | GQA (80->8) |
| RoPE theta | 1M | 100K | 500K |
| MoE layers | Alternating | No | No |
| Audio input | Built-in | No | No |
| Talker head | Yes | No | No |
