//! Talker: Speech synthesis MoE model for Qwen3-Omni
//!
//! Takes talker tokens from the Thinker and generates multi-codebook
//! audio tokens that can be decoded by Code2Wav.
//!
//! Architecture: 3B MoE with 0.3B active parameters, outputs 4 parallel
//! codebook streams.

use super::config::TalkerConfig;
use crate::models::with_tracing::{linear_no_bias, Linear, RmsNorm};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use std::sync::Arc;

/// Rotary embedding for Talker
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq: usize, device: &Device, dtype: DType) -> Result<Self> {
        let theta = 10000.0f32;
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / theta.powf(2.0 * i as f32 / dim as f32))
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

/// Grouped Query Attention for Talker
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
    fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
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

        let q = q
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, offset)?;

        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let repeat = self.num_heads / self.num_kv_heads;
        let k = k.repeat(&[1, repeat, 1, 1])?;
        let v = v.repeat(&[1, repeat, 1, 1])?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut attn = (q.matmul(&k.t()?)? * scale)?;

        if let Some(mask) = mask {
            attn = attn.broadcast_add(mask)?;
        }

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, seq, ()))?;

        self.o_proj.forward(&out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// MLP Expert
struct MLPExpert {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl MLPExpert {
    fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
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

/// Sparse MoE block
struct SparseMoE {
    gate: Linear,
    experts: Vec<MLPExpert>,
    num_experts_per_tok: usize,
}

impl SparseMoE {
    fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
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
        })
    }
}

impl Module for SparseMoE {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden))?;

        let router_logits = self.gate.forward(&xs_flat)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        let top_indices = routing_weights
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, self.num_experts_per_tok)?
            .contiguous()?;
        let top_weights = routing_weights.gather(&top_indices, D::Minus1)?;

        // Normalize
        let sum = top_weights.sum_keepdim(D::Minus1)?;
        let top_weights = top_weights.broadcast_div(&sum)?;

        let routing_weights_vec = top_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let expert_indices_vec = top_indices.to_vec2::<u32>()?;

        let mut result = xs_flat.zeros_like()?;

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

/// Decoder layer for Talker
struct DecoderLayer {
    self_attn: Attention,
    moe: SparseMoE,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            moe: SparseMoE::new(cfg, vb.pp("mlp"))?,
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
        let xs = self.moe.forward(&xs)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Multi-codebook output head
#[allow(dead_code)]
struct CodebookHead {
    heads: Vec<Linear>,
    num_codebooks: usize,
    codebook_size: usize,
}

impl CodebookHead {
    fn new(cfg: &TalkerConfig, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let mut heads = Vec::with_capacity(cfg.num_codebooks);
        let vb_heads = vb.pp("heads");
        for i in 0..cfg.num_codebooks {
            heads.push(linear_no_bias(hidden_size, cfg.codebook_size, vb_heads.pp(i))?);
        }

        Ok(Self {
            heads,
            num_codebooks: cfg.num_codebooks,
            codebook_size: cfg.codebook_size,
        })
    }

    /// Generate codec tokens for all codebooks
    /// Returns [batch, seq, num_codebooks]
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let mut codebook_tokens = Vec::with_capacity(self.num_codebooks);

        for head in &self.heads {
            let logits = head.forward(hidden)?;
            let tokens = logits.argmax(D::Minus1)?;
            codebook_tokens.push(tokens);
        }

        Tensor::stack(&codebook_tokens, 2)
    }
}

/// Talker: Speech synthesis model
pub struct Talker {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    codebook_head: CodebookHead,
    rotary: Arc<RotaryEmbedding>,
    device: Device,
    dtype: DType,
}

impl Talker {
    pub fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Talker tokens vocabulary (from Thinker output)
        let talker_vocab_size = 4096;
        let embed_tokens = candle_nn::embedding(talker_vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("model.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model.norm"))?;
        let codebook_head = CodebookHead::new(cfg, cfg.hidden_size, vb.pp("codebook_head"))?;

        let rotary = Arc::new(RotaryEmbedding::new(cfg.head_dim, 8192, &device, dtype)?);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            codebook_head,
            rotary,
            device,
            dtype,
        })
    }

    /// Forward pass: convert talker tokens to codec tokens
    ///
    /// # Arguments
    /// * `talker_tokens` - Tokens from Thinker [batch, seq]
    ///
    /// # Returns
    /// * Codec tokens for all codebooks [batch, seq, num_codebooks]
    pub fn forward(&mut self, talker_tokens: &Tensor) -> Result<Tensor> {
        let embeddings = self.embed_tokens.forward(talker_tokens)?;
        let seq_len = embeddings.dim(1)?;
        let batch = embeddings.dim(0)?;

        let mask = self.causal_mask(batch, seq_len, 0)?;

        let mut hidden = embeddings;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), 0)?;
        }

        hidden = self.norm.forward(&hidden)?;

        self.codebook_head.forward(&hidden)
    }

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
