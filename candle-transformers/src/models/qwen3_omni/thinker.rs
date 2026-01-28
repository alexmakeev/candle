//! Thinker: Main reasoning MoE model for Qwen3-Omni
//!
//! This is the "brain" of Qwen3-Omni - a 30B MoE model with 3B active parameters
//! that processes audio tokens + text and generates both text responses
//! and "talker tokens" for speech synthesis.
//!
//! Based on Qwen3 MoE architecture with additional audio understanding.

use super::config::ThinkerConfig;
use crate::models::with_tracing::{linear_no_bias, Linear, RmsNorm};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

/// Output from Thinker forward pass
pub struct ThinkerOutput {
    /// Generated text token logits [batch, seq, vocab]
    pub text_logits: Tensor,

    /// Talker tokens for speech synthesis [batch, seq]
    pub talker_tokens: Tensor,

    /// Hidden states (for debugging/analysis)
    pub hidden_states: Tensor,
}

/// Rotary positional embedding
#[allow(dead_code)]
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    dim: usize,
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
            dim,
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

/// Grouped Query Attention
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(cfg: &ThinkerConfig, vb: VarBuilder) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim;

        Ok(Self {
            q_proj: linear_no_bias(hidden, n_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(hidden, n_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(hidden, n_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(n_heads * head_dim, hidden, vb.pp("o_proj"))?,
            num_heads: n_heads,
            num_kv_heads: n_kv_heads,
            head_dim,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape for attention
        let q = q
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, offset)?;

        // Handle KV cache
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Repeat KV for GQA
        let repeat = self.num_heads / self.num_kv_heads;
        let k = k.repeat(&[1, repeat, 1, 1])?;
        let v = v.repeat(&[1, repeat, 1, 1])?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn = (q.matmul(&k.t()?)? * scale)?;

        if let Some(mask) = mask {
            attn = attn.broadcast_add(mask)?;
        }

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // Reshape and project
        let out = out.transpose(1, 2)?.reshape((b, seq, ()))?;
        self.o_proj.forward(&out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// MLP Expert for MoE
struct MLPExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLPExpert {
    fn new(cfg: &ThinkerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.moe_intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.moe_intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.moe_intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLPExpert {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Dense MLP (for non-MoE layers)
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLP {
    fn new(cfg: &ThinkerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Sparse MoE block
struct SparseMoE {
    gate: Linear,
    experts: Vec<MLPExpert>,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
}

impl SparseMoE {
    fn new(cfg: &ThinkerConfig, vb: VarBuilder) -> Result<Self> {
        let gate = linear_no_bias(cfg.hidden_size, cfg.num_experts, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(cfg.num_experts);
        let vb_experts = vb.pp("experts");
        for i in 0..cfg.num_experts {
            experts.push(MLPExpert::new(cfg, vb_experts.pp(i))?);
        }

        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        })
    }
}

impl Module for SparseMoE {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden))?;

        // Compute router logits and weights
        let router_logits = self.gate.forward(&xs_flat)?;
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

        // Process through experts (naive implementation)
        let routing_weights_vec = top_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let expert_indices_vec = top_indices.to_vec2::<u32>()?;

        let mut result = xs_flat.zeros_like()?;

        // Group tokens by expert
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

            let indices = Tensor::from_slice(&token_indices, token_indices.len(), xs.device())?;
            let weights = Tensor::from_slice(&weights, weights.len(), xs.device())?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;

            let selected = xs_flat.index_select(&indices, 0)?;
            let expert_out = self.experts[expert_idx].forward(&selected)?;
            let weighted = expert_out.broadcast_mul(&weights)?;

            result = result.index_add(&indices, &weighted, 0)?;
        }

        result.reshape((b, seq, hidden))
    }
}

/// Feed-forward: either MLP or MoE
enum FeedForward {
    Dense(MLP),
    Sparse(SparseMoE),
}

impl FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Sparse(moe) => moe.forward(xs),
        }
    }
}

/// Decoder layer
struct DecoderLayer {
    self_attn: Attention,
    ffn: FeedForward,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(layer_idx: usize, cfg: &ThinkerConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = Attention::new(cfg, vb.pp("self_attn"))?;

        // Use MoE for every decoder_sparse_step layer
        let ffn = if cfg.num_experts > 0
            && cfg.decoder_sparse_step > 0
            && (layer_idx + 1) % cfg.decoder_sparse_step == 0
        {
            FeedForward::Sparse(SparseMoE::new(cfg, vb.pp("mlp"))?)
        } else {
            FeedForward::Dense(MLP::new(cfg, vb.pp("mlp"))?)
        };

        Ok(Self {
            self_attn,
            ffn,
            ln1: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, rotary, mask, offset)?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.ln2.forward(&xs)?;
        let xs = self.ffn.forward(&xs)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Thinker: Main reasoning model
pub struct Thinker {
    embed_tokens: candle_nn::Embedding,
    audio_embed: Option<Linear>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    talker_head: Option<Linear>,
    rotary: Arc<RotaryEmbedding>,
    device: Device,
    dtype: DType,
}

impl Thinker {
    pub fn new(cfg: &ThinkerConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let t0 = std::time::Instant::now();

        eprintln!("[THINKER] Loading embed_tokens (vocab={}, hidden={})...", cfg.vocab_size, cfg.hidden_size);
        let embed_tokens = candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        eprintln!("[THINKER] embed_tokens loaded in {:?}", t0.elapsed());

        // audio_embed is optional - only present in audio-capable models
        let audio_embed = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("model.audio_embed")).ok();
        eprintln!("[THINKER] audio_embed: {}", if audio_embed.is_some() { "loaded" } else { "skipped (optional)" });

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        eprintln!("[THINKER] Loading {} decoder layers...", cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let lt = std::time::Instant::now();
            layers.push(DecoderLayer::new(i, cfg, vb_layers.pp(i))?);
            if i % 8 == 0 || i == cfg.num_hidden_layers - 1 {
                eprintln!("[THINKER] Layer {}/{} loaded in {:?} (total: {:?})", i + 1, cfg.num_hidden_layers, lt.elapsed(), t0.elapsed());
            }
        }

        eprintln!("[THINKER] Loading norm layer...");
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;

        eprintln!("[THINKER] Loading lm_head (tie_embeddings={})...", cfg.tie_word_embeddings);
        let lm_head = if cfg.tie_word_embeddings {
            Linear::from_weights(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };

        // Talker head: project to talker token space (optional)
        let talker_head = linear_no_bias(cfg.hidden_size, 4096, vb.pp("talker_head")).ok();
        eprintln!("[THINKER] talker_head: {}", if talker_head.is_some() { "loaded" } else { "skipped (optional)" });

        let rotary = Arc::new(RotaryEmbedding::new(cfg, &device, dtype)?);
        eprintln!("[THINKER] Model loaded in {:?}", t0.elapsed());

        Ok(Self {
            embed_tokens,
            audio_embed,
            layers,
            norm,
            lm_head,
            talker_head,
            rotary,
            device,
            dtype,
        })
    }

    /// Forward pass with audio tokens and optional text prompt
    pub fn forward(
        &mut self,
        audio_tokens: &Tensor,
        text_prompt: Option<&Tensor>,
    ) -> Result<ThinkerOutput> {
        // Embed audio tokens through audio embedding
        let audio_embeds = self.embed_tokens.forward(audio_tokens)?;
        let audio_embeds = match &self.audio_embed {
            Some(embed) => embed.forward(&audio_embeds)?,
            None => audio_embeds,
        };

        // Combine with text prompt if present
        let (embeddings, total_len) = match text_prompt {
            Some(text) => {
                let text_embeds = self.embed_tokens.forward(text)?;
                let combined = Tensor::cat(&[audio_embeds, text_embeds], 1)?;
                let len = combined.dim(1)?;
                (combined, len)
            }
            None => {
                let len = audio_embeds.dim(1)?;
                (audio_embeds, len)
            }
        };

        // Create causal mask
        let mask = self.causal_mask(embeddings.dim(0)?, total_len, 0)?;

        // Process through layers
        let mut hidden = embeddings;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), 0)?;
        }

        hidden = self.norm.forward(&hidden)?;

        // Generate outputs
        let text_logits = self.lm_head.forward(&hidden)?;
        let talker_tokens = match &self.talker_head {
            Some(head) => head.forward(&hidden)?.argmax(D::Minus1)?,
            None => Tensor::zeros((hidden.dim(0)?, hidden.dim(1)?), DType::U32, hidden.device())?,
        };

        Ok(ThinkerOutput {
            text_logits,
            talker_tokens,
            hidden_states: hidden,
        })
    }

    /// Forward pass for text-only input
    pub fn forward_text_only(&mut self, text_tokens: &Tensor) -> Result<ThinkerOutput> {
        let embeddings = self.embed_tokens.forward(text_tokens)?;
        let seq_len = embeddings.dim(1)?;
        let batch = embeddings.dim(0)?;

        let mask = self.causal_mask(batch, seq_len, 0)?;

        let mut hidden = embeddings;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), 0)?;
        }

        hidden = self.norm.forward(&hidden)?;

        let text_logits = self.lm_head.forward(&hidden)?;
        let talker_tokens = match &self.talker_head {
            Some(head) => head.forward(&hidden)?.argmax(D::Minus1)?,
            None => Tensor::zeros((batch, seq_len), DType::U32, hidden.device())?,
        };

        Ok(ThinkerOutput {
            text_logits,
            talker_tokens,
            hidden_states: hidden,
        })
    }

    /// Generate text tokens autoregressively
    pub fn generate(
        &mut self,
        audio_tokens: &Tensor,
        text_prompt: Option<&Tensor>,
        max_tokens: usize,
    ) -> Result<Tensor> {
        // Initial forward pass
        let output = self.forward(audio_tokens, text_prompt)?;
        let mut next_token = output.text_logits.narrow(1, output.text_logits.dim(1)? - 1, 1)?;
        next_token = next_token.argmax(D::Minus1)?;

        let mut tokens = vec![next_token.clone()];
        let mut offset = audio_tokens.dim(1)? + text_prompt.map(|t| t.dim(1).unwrap_or(0)).unwrap_or(0);

        for _ in 1..max_tokens {
            let embeddings = self.embed_tokens.forward(&next_token)?;

            let mut hidden = embeddings;
            for layer in &mut self.layers {
                hidden = layer.forward(&hidden, &self.rotary, None, offset)?;
            }

            hidden = self.norm.forward(&hidden)?;
            let logits = self.lm_head.forward(&hidden)?;
            next_token = logits.argmax(D::Minus1)?;

            // Check for EOS token (assuming 2 is EOS)
            let token_val: u32 = next_token.squeeze(0)?.squeeze(0)?.to_scalar()?;
            if token_val == 2 {
                break;
            }

            tokens.push(next_token.clone());
            offset += 1;
        }

        Tensor::cat(&tokens, 1)
    }

    /// Create causal attention mask
    fn causal_mask(&self, _batch: usize, seq: usize, offset: usize) -> Result<Tensor> {
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

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}
