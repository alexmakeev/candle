//! Quantized Thinker for Qwen3-Omni
//!
//! Main reasoning MoE model with 48 layers and 128 experts.
//!
//! GGUF tensor structure:
//! - thinker.model.embed_tokens.weight: [152064, 2048] Q8_0
//! - thinker.model.layers.{N}.self_attn.{q,k,v,o}_proj.weight: Q8_0
//! - thinker.model.layers.{N}.self_attn.{q,k}_norm.weight: F32
//! - thinker.model.layers.{N}.mlp.gate.weight: [128, 2048] Q8_0 (router)
//! - thinker.model.layers.{N}.mlp.experts.{0-127}.{gate,up,down}_proj.weight: Q8_0
//! - thinker.model.layers.{N}.{input,post_attention}_layernorm.weight: F32
//! - thinker.model.norm.weight: F32
//! - thinker.lm_head.weight: [152064, 2048] Q8_0

use super::config::ThinkerConfig;
use super::gguf_loader::Gguf;
use crate::models::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use candle::quantized::QTensor;
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::kv_cache::ConcatKvCache;
use std::io::{Read, Seek};
use std::sync::Arc;

/// Output from Thinker forward pass
pub struct ThinkerOutput {
    /// Text token logits [batch, seq, vocab]
    pub text_logits: Tensor,
    /// Hidden states for Talker [batch, seq, hidden]
    pub hidden_states: Tensor,
}

/// Rotary positional embedding
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(cfg: &ThinkerConfig, device: &Device, dtype: DType) -> Result<Self> {
        let dim = cfg.head_dim;
        let theta = cfg.rope_theta;
        let max_seq = cfg.max_position_embeddings;

        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_slice(&inv_freq, dim / 2, device)?;

        let positions: Vec<f32> = (0..max_seq).map(|i| i as f32).collect();
        let positions = Tensor::from_slice(&positions, max_seq, device)?;

        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;

        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;

        let q_rot = apply_rotary(q, &cos, &sin)?;
        let k_rot = apply_rotary(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, _seq, dim) = x.dims4()?;

    let x1 = x.narrow(D::Minus1, 0, dim / 2)?;
    let x2 = x.narrow(D::Minus1, dim / 2, dim / 2)?;

    let cos = cos.narrow(1, 0, dim / 2)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.narrow(1, 0, dim / 2)?.unsqueeze(0)?.unsqueeze(0)?;

    let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let r2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;

    Tensor::cat(&[r1, r2], D::Minus1)
}

/// Repeat KV heads for GQA
fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(x.clone())
    } else {
        let (b, h, seq, d) = x.dims4()?;
        x.unsqueeze(2)?
            .expand((b, h, n_rep, seq, d))?
            .reshape((b, h * n_rep, seq, d))
    }
}

/// Grouped Query Attention with QK norms
struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: ConcatKvCache,
    dtype: DType,
}

impl Attention {
    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;
        let in_dtype = xs.dtype();

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape for multi-head
        let q = q.reshape((b, seq, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq, self.num_kv_heads, self.head_dim))?;
        let v = v
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK norms (per-head normalization)
        let q = q.flatten(0, 1)?; // [B*seq, heads, dim] -> flatten for norm
        let k = k.flatten(0, 1)?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;
        let q = q.reshape((b, seq, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, seq, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        // Convert to compute dtype
        let q = q.to_dtype(self.dtype)?;
        let k = k.to_dtype(self.dtype)?;
        let v = v.to_dtype(self.dtype)?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, offset)?;

        // KV cache
        let (k, v) = self.kv_cache.append(&k, &v)?;

        // Repeat KV for GQA
        let n_rep = self.num_heads / self.num_kv_heads;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn = (q.matmul(&k.t()?)? * scale)?;

        if let Some(m) = mask {
            attn = attn.broadcast_add(m)?;
        }

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // Reshape and project
        let out = out.transpose(1, 2)?.reshape((b, seq, ()))?;
        self.o_proj.forward(&out.to_dtype(in_dtype)?)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset();
    }
}

/// Single MoE expert
struct MlpExpert {
    gate_proj: Arc<QTensor>,
    up_proj: Arc<QTensor>,
    down_proj: Arc<QTensor>,
}

impl MlpExpert {
    fn forward(&self, xs: &Tensor, device: &Device) -> Result<Tensor> {
        let gate = self.gate_proj.dequantize(device)?;
        let up = self.up_proj.dequantize(device)?;
        let down = self.down_proj.dequantize(device)?;

        // xs @ gate.T
        let gate_out = xs.matmul(&gate.t()?)?;
        let gate_out = candle_nn::ops::silu(&gate_out)?;

        // xs @ up.T
        let up_out = xs.matmul(&up.t()?)?;

        // (gate * up) @ down.T
        let hidden = (gate_out * up_out)?;
        hidden.matmul(&down.t()?)
    }
}

/// Sparse MoE layer with top-k routing
struct SparseMoE {
    gate: QMatMul,
    experts: Vec<MlpExpert>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    device: Device,
}

impl SparseMoE {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden))?;
        let _num_tokens = xs_flat.dim(0)?;
        let original_dtype = xs_flat.dtype();

        // Convert to F32 for routing computation
        let xs_f32 = if original_dtype != DType::F32 {
            xs_flat.to_dtype(DType::F32)?
        } else {
            xs_flat.clone()
        };

        // Compute router logits and weights
        let router_logits = self.gate.forward(&xs_f32)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Get top-k experts
        let top_indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let top_weights = routing_weights.gather(&top_indices, D::Minus1)?;

        // Normalize weights if needed
        let top_weights = if self.norm_topk_prob {
            let sum = top_weights.sum_keepdim(D::Minus1)?;
            top_weights.broadcast_div(&sum)?
        } else {
            top_weights
        };

        // Get routing info
        let routing_weights_vec = top_weights.to_vec2::<f32>()?;
        let expert_indices_vec = top_indices.to_vec2::<u32>()?;

        // Initialize result
        let mut result = xs_f32.zeros_like()?;

        // Group tokens by expert for batched processing
        let mut expert_to_tokens: Vec<Vec<(usize, f32)>> = vec![vec![]; self.experts.len()];
        for (token_idx, (weights, experts)) in routing_weights_vec
            .iter()
            .zip(expert_indices_vec.iter())
            .enumerate()
        {
            for (&w, &e) in weights.iter().zip(experts.iter()) {
                expert_to_tokens[e as usize].push((token_idx, w));
            }
        }

        // Process each expert
        for (expert_idx, token_info) in expert_to_tokens.iter().enumerate() {
            if token_info.is_empty() {
                continue;
            }

            let token_indices: Vec<u32> = token_info.iter().map(|&(i, _)| i as u32).collect();
            let weights: Vec<f32> = token_info.iter().map(|&(_, w)| w).collect();

            let indices = Tensor::from_slice(&token_indices, token_indices.len(), &self.device)?;
            let weights = Tensor::from_slice(&weights, weights.len(), &self.device)?
                .reshape(((), 1))?;

            let selected = xs_f32.index_select(&indices, 0)?;
            let expert_out = self.experts[expert_idx].forward(&selected, &self.device)?;
            let weighted = expert_out.broadcast_mul(&weights)?;

            result = result.index_add(&indices, &weighted, 0)?;
        }

        // Convert back to original dtype
        let result = if original_dtype != DType::F32 {
            result.to_dtype(original_dtype)?
        } else {
            result
        };

        result.reshape((b, seq, hidden))
    }
}

/// Decoder layer with MoE
struct DecoderLayer {
    self_attn: Attention,
    moe: SparseMoE,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        // Attention with residual
        let residual = xs;
        let xs = self.ln1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, rotary, mask, offset)?;
        let xs = (residual + xs)?;

        // MoE with residual
        let residual = &xs;
        let xs = self.ln2.forward(&xs)?;
        let xs = self.moe.forward(&xs)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Quantized Thinker model
pub struct Thinker {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QMatMul,
    rotary: Arc<RotaryEmbedding>,
    dtype: DType,
    device: Device,
}

impl Thinker {
    /// Load from GGUF file
    pub fn from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &ThinkerConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = "thinker.model";
        let dtype = DType::F32;

        // Token embeddings
        let embed_tokens = gg.embedding(
            &format!("{prefix}.embed_tokens.weight"),
            cfg.hidden_size,
        )?;

        // Decoder layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{i}");

            // Attention
            let self_attn = Attention {
                q_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.q_proj.weight"))?,
                k_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.k_proj.weight"))?,
                v_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.v_proj.weight"))?,
                o_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.o_proj.weight"))?,
                q_norm: gg.rms_norm(
                    &format!("{layer_prefix}.self_attn.q_norm.weight"),
                    cfg.rms_norm_eps,
                )?,
                k_norm: gg.rms_norm(
                    &format!("{layer_prefix}.self_attn.k_norm.weight"),
                    cfg.rms_norm_eps,
                )?,
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                kv_cache: ConcatKvCache::new(2),
                dtype,
            };

            // MoE router gate
            let gate = gg.qmatmul(&format!("{layer_prefix}.mlp.gate.weight"))?;

            // MoE experts
            let mut experts = Vec::with_capacity(cfg.num_experts);
            for e in 0..cfg.num_experts {
                let expert_prefix = format!("{layer_prefix}.mlp.experts.{e}");
                experts.push(MlpExpert {
                    gate_proj: Arc::new(gg.tensor(&format!("{expert_prefix}.gate_proj.weight"))?),
                    up_proj: Arc::new(gg.tensor(&format!("{expert_prefix}.up_proj.weight"))?),
                    down_proj: Arc::new(gg.tensor(&format!("{expert_prefix}.down_proj.weight"))?),
                });
            }

            let moe = SparseMoE {
                gate,
                experts,
                num_experts_per_tok: cfg.num_experts_per_tok,
                norm_topk_prob: cfg.norm_topk_prob,
                device: device.clone(),
            };

            // Layer norms
            let ln1 = gg.rms_norm(
                &format!("{layer_prefix}.input_layernorm.weight"),
                cfg.rms_norm_eps,
            )?;
            let ln2 = gg.rms_norm(
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )?;

            layers.push(DecoderLayer { self_attn, moe, ln1, ln2 });
        }

        // Final norm
        let norm = gg.rms_norm(&format!("{prefix}.norm.weight"), cfg.rms_norm_eps)?;

        // LM head
        let lm_head = gg.qmatmul("thinker.lm_head.weight")?;

        // Rotary embedding
        let rotary = Arc::new(RotaryEmbedding::new(cfg, device, dtype)?);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            dtype,
            device: device.clone(),
        })
    }

    /// Create causal attention mask
    fn causal_mask(&self, seq: usize, offset: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq)
            .flat_map(|i| {
                (0..seq + offset).map(move |j| {
                    if j <= i + offset {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Tensor::from_slice(&mask, (1, 1, seq, seq + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// Forward pass with audio embeddings
    pub fn forward(
        &mut self,
        audio_embeds: &Tensor,
        text_tokens: Option<&Tensor>,
    ) -> Result<ThinkerOutput> {
        // Combine audio embeddings with text embeddings if present
        let (embeddings, total_len) = match text_tokens {
            Some(text) => {
                let text_embeds = self.embed_tokens.forward(text)?;
                let combined = Tensor::cat(&[audio_embeds, &text_embeds], 1)?;
                let len = combined.dim(1)?;
                (combined, len)
            }
            None => {
                let len = audio_embeds.dim(1)?;
                (audio_embeds.clone(), len)
            }
        };

        // Create causal mask
        let mask = self.causal_mask(total_len, 0)?;

        // Process through layers
        let mut hidden = embeddings;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), 0)?;
        }

        hidden = self.norm.forward(&hidden)?;

        // Generate text logits
        let text_logits = self.lm_head.forward(&hidden)?;

        Ok(ThinkerOutput {
            text_logits,
            hidden_states: hidden,
        })
    }

    /// Forward pass for text-only input
    pub fn forward_text_only(&mut self, text_tokens: &Tensor) -> Result<ThinkerOutput> {
        let embeddings = self.embed_tokens.forward(text_tokens)?;
        let seq_len = embeddings.dim(1)?;

        let mask = self.causal_mask(seq_len, 0)?;

        let mut hidden = embeddings;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), 0)?;
        }

        hidden = self.norm.forward(&hidden)?;

        let text_logits = self.lm_head.forward(&hidden)?;

        Ok(ThinkerOutput {
            text_logits,
            hidden_states: hidden,
        })
    }

    /// Generate next token (for incremental decoding)
    pub fn forward_one_token(
        &mut self,
        token: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let embeddings = self.embed_tokens.forward(token)?;

        let mut hidden = embeddings;
        for layer in &mut self.layers {
            // No mask needed for single token with KV cache
            hidden = layer.forward(&hidden, &self.rotary, None, offset)?;
        }

        hidden = self.norm.forward(&hidden)?;
        self.lm_head.forward(&hidden)
    }

    /// Clear KV cache
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
