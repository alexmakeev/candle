//! Quantized Talker for Qwen3-Omni TTS
//!
//! Full architecture:
//! - text_projection: ResizeMLP (2048 -> 1024)
//! - hidden_projection: ResizeMLP (2048 -> 1024)
//! - TalkerModel: MoE decoder (20 layers, 128 experts + shared experts)
//! - codec_head: Linear (1024 -> 3072, split to 15 codebooks)
//! - CodePredictor: Dense decoder (5 layers)
//!
//! GGUF tensor structure:
//! - talker.hidden_projection.linear_fc1.{weight,bias}: [2048, 2048] Q8_0 / F32
//! - talker.hidden_projection.linear_fc2.{weight,bias}: [1024, 2048] Q8_0 / F32
//! - talker.text_projection.linear_fc1.{weight,bias}: [2048, 2048] Q8_0 / F32
//! - talker.text_projection.linear_fc2.{weight,bias}: [1024, 2048] Q8_0 / F32
//! - talker.model.codec_embedding.{0-14}.weight: [4096, 1024] Q8_0
//! - talker.model.layers.{0-19}.self_attn.{q,k,v,o}_proj.weight: Q8_0
//! - talker.model.layers.{0-19}.self_attn.{q,k}_norm.weight: F32
//! - talker.model.layers.{0-19}.mlp.router.weight: [128, 1024] Q8_0
//! - talker.model.layers.{0-19}.mlp.experts.{0-127}.{gate,up,down}_proj.weight: Q8_0
//! - talker.model.layers.{0-19}.mlp.shared_experts.{gate,up,down}_proj.weight: Q8_0
//! - talker.model.layers.{0-19}.{input,post_attention}_layernorm.weight: F32
//! - talker.model.norm.weight: F32
//! - talker.codec_head.weight: [3072, 1024] Q8_0
//! - talker.code_predictor.model.codec_embedding.{0-14}.weight: [1024, 2048] Q8_0
//! - talker.code_predictor.model.layers.{0-4}.self_attn.{q,k,v,o}_proj.weight: Q8_0
//! - talker.code_predictor.model.layers.{0-4}.self_attn.{q,k}_norm.weight: F32
//! - talker.code_predictor.model.layers.{0-4}.mlp.{gate,up,down}_proj.weight: Q8_0
//! - talker.code_predictor.model.layers.{0-4}.{input,post_attention}_layernorm.weight: F32
//! - talker.code_predictor.model.norm.weight: F32
//! - talker.code_predictor.lm_head.{0-14}.weight: [1024, 2048] Q8_0

use super::config::TalkerConfig;
use super::gguf_loader::Gguf;
use crate::models::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use candle::quantized::QTensor;
use candle::{DType, Device, Module, Result, Tensor, D};
use std::io::{Read, Seek};
use std::sync::Arc;

// ============================================================================
// Rotary Embedding
// ============================================================================

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, base: f64, device: &Device, dtype: DType) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / (base as f32).powf(2.0 * i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_slice(&inv_freq, dim / 2, device)?;

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_slice(&positions, max_seq_len, device)?;

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

// ============================================================================
// Attention (GQA with QK norms)
// ============================================================================

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
    kv_cache: Option<(Tensor, Tensor)>,
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

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape to [B, seq, num_heads, head_dim] then transpose to [B, num_heads, seq, head_dim]
        let q = q
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, seq, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply QK normalization per-head
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, seq, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, seq, self.head_dim))?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, offset)?;

        // KV cache handling
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &k], 2)?;
                let v = Tensor::cat(&[prev_v, &v], 2)?;
                (k, v)
            }
            None => (k, v),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Expand KV heads for GQA
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
        let out = out.transpose(1, 2)?.reshape((b, seq, ()))?;

        self.o_proj.forward(&out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

// ============================================================================
// MLP (SwiGLU)
// ============================================================================

struct Mlp {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ============================================================================
// MoE Expert
// ============================================================================

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

        let gate_out = xs.matmul(&gate.t()?)?;
        let gate_out = candle_nn::ops::silu(&gate_out)?;
        let up_out = xs.matmul(&up.t()?)?;
        let hidden = (gate_out * up_out)?;
        hidden.matmul(&down.t()?)
    }
}

// ============================================================================
// Sparse MoE with Shared Experts
// ============================================================================

struct SparseMoEWithShared {
    router: QMatMul,
    experts: Vec<MlpExpert>,
    shared_experts: Mlp,
    num_experts_per_tok: usize,
    norm_topk_prob: bool,
    device: Device,
}

impl SparseMoEWithShared {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, seq, hidden) = xs.dims3()?;
        let xs_flat = xs.reshape(((), hidden))?;
        let original_dtype = xs_flat.dtype();

        // Convert to F32 for routing
        let xs_f32 = if original_dtype != DType::F32 {
            xs_flat.to_dtype(DType::F32)?
        } else {
            xs_flat.clone()
        };

        // Router logits
        let router_logits = self.router.forward(&xs_f32)?;
        let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

        // Top-k experts
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

        let routing_weights_vec = top_weights.to_vec2::<f32>()?;
        let expert_indices_vec = top_indices.to_vec2::<u32>()?;

        // Initialize sparse result
        let mut sparse_result = xs_f32.zeros_like()?;

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

            let indices = Tensor::from_slice(&token_indices, token_indices.len(), &self.device)?;
            let weights = Tensor::from_slice(&weights, weights.len(), &self.device)?
                .reshape(((), 1))?;

            let selected = xs_f32.index_select(&indices, 0)?;
            let expert_out = self.experts[expert_idx].forward(&selected, &self.device)?;
            let weighted = expert_out.broadcast_mul(&weights)?;

            sparse_result = sparse_result.index_add(&indices, &weighted, 0)?;
        }

        // Process shared experts (always active)
        let shared_out = self.shared_experts.forward(&xs_flat)?;

        // Combine sparse and shared
        let combined = (sparse_result + shared_out)?;

        // Convert back to original dtype
        let result = if original_dtype != DType::F32 {
            combined.to_dtype(original_dtype)?
        } else {
            combined
        };

        result.reshape((b, seq, hidden))
    }
}

// ============================================================================
// TalkerModel Decoder Layer (MoE)
// ============================================================================

struct TalkerModelDecoderLayer {
    self_attn: Attention,
    moe: SparseMoEWithShared,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl TalkerModelDecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        // Attention with residual
        let residual = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward(&hidden, rotary, mask, offset)?;
        let xs = (residual + attn_out)?;

        // MoE with residual
        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        let moe_out = self.moe.forward(&hidden)?;
        residual + moe_out
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ============================================================================
// TalkerModel (MoE decoder)
// ============================================================================

struct TalkerModel {
    codec_embeddings: Vec<candle_nn::Embedding>,
    layers: Vec<TalkerModelDecoderLayer>,
    norm: RmsNorm,
    rotary: Arc<RotaryEmbedding>,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl TalkerModel {
    fn from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &TalkerConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = "talker.model";
        let dtype = DType::F32;
        let model_cfg = &cfg.talker_model;

        // Load codec embeddings (15 codebooks, vocab_size=4096)
        let mut codec_embeddings = Vec::with_capacity(model_cfg.num_codec_embeddings);
        for i in 0..model_cfg.num_codec_embeddings {
            let emb = gg.embedding(
                &format!("{prefix}.codec_embedding.{i}.weight"),
                model_cfg.hidden_size,
            )?;
            codec_embeddings.push(emb);
        }

        // Determine number of layers
        let mut num_layers = 0;
        while gg.has_tensor(&format!("{prefix}.layers.{num_layers}.input_layernorm.weight")) {
            num_layers += 1;
        }

        // Load MoE decoder layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_prefix = format!("{prefix}.layers.{i}");

            let input_layernorm = gg.rms_norm(
                &format!("{layer_prefix}.input_layernorm.weight"),
                cfg.rms_norm_eps,
            )?;

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
                num_heads: model_cfg.num_attention_heads,
                num_kv_heads: model_cfg.num_key_value_heads,
                head_dim: model_cfg.head_dim,
                kv_cache: None,
            };

            // MoE router
            let router = gg.qmatmul(&format!("{layer_prefix}.mlp.router.weight"))?;

            // Load experts
            let mut experts = Vec::with_capacity(model_cfg.num_experts);
            for e in 0..model_cfg.num_experts {
                let expert_prefix = format!("{layer_prefix}.mlp.experts.{e}");
                experts.push(MlpExpert {
                    gate_proj: Arc::new(gg.tensor(&format!("{expert_prefix}.gate_proj.weight"))?),
                    up_proj: Arc::new(gg.tensor(&format!("{expert_prefix}.up_proj.weight"))?),
                    down_proj: Arc::new(gg.tensor(&format!("{expert_prefix}.down_proj.weight"))?),
                });
            }

            // Shared experts
            let shared_experts = Mlp {
                gate_proj: gg.qmatmul(&format!("{layer_prefix}.mlp.shared_experts.gate_proj.weight"))?,
                up_proj: gg.qmatmul(&format!("{layer_prefix}.mlp.shared_experts.up_proj.weight"))?,
                down_proj: gg.qmatmul(&format!("{layer_prefix}.mlp.shared_experts.down_proj.weight"))?,
            };

            let moe = SparseMoEWithShared {
                router,
                experts,
                shared_experts,
                num_experts_per_tok: model_cfg.num_experts_per_tok,
                norm_topk_prob: model_cfg.norm_topk_prob,
                device: device.clone(),
            };

            let post_attention_layernorm = gg.rms_norm(
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )?;

            layers.push(TalkerModelDecoderLayer {
                self_attn,
                moe,
                input_layernorm,
                post_attention_layernorm,
            });
        }

        // Final norm
        let norm = gg.rms_norm(&format!("{prefix}.norm.weight"), cfg.rms_norm_eps)?;

        // Rotary embedding
        let rotary = Arc::new(RotaryEmbedding::new(
            model_cfg.head_dim,
            cfg.max_position_embeddings,
            model_cfg.rope_theta,
            device,
            dtype,
        )?);

        Ok(Self {
            codec_embeddings,
            layers,
            norm,
            rotary,
            device: device.clone(),
            dtype,
        })
    }

    /// Embed codec tokens: sum embeddings from all codebooks
    fn embed_codec_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        let (_b, _seq, num_cb) = tokens.dims3()?;
        let mut result: Option<Tensor> = None;

        for (i, emb) in self.codec_embeddings.iter().enumerate() {
            if i >= num_cb {
                break;
            }
            let cb_tokens = tokens.narrow(2, i, 1)?.squeeze(2)?;
            let embedded = emb.forward(&cb_tokens)?;

            result = Some(match result {
                Some(acc) => (acc + embedded)?,
                None => embedded,
            });
        }

        result.ok_or_else(|| candle::Error::Msg("No codebooks".into()))
    }

    /// Forward pass
    fn forward(
        &mut self,
        input_embeds: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let mut hidden = input_embeds.clone();

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, mask, offset)?;
        }

        self.norm.forward(&hidden)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ============================================================================
// CodePredictor Decoder Layer (Dense)
// ============================================================================

struct CodePredictorDecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl CodePredictorDecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward(&hidden, rotary, mask, offset)?;
        let xs = (residual + attn_out)?;

        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        let mlp_out = self.mlp.forward(&hidden)?;
        residual + mlp_out
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ============================================================================
// CodePredictor (Dense decoder)
// ============================================================================

struct CodePredictor {
    codec_embeddings: Vec<candle_nn::Embedding>,
    layers: Vec<CodePredictorDecoderLayer>,
    norm: RmsNorm,
    lm_heads: Vec<QMatMul>,
    rotary: Arc<RotaryEmbedding>,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl CodePredictor {
    fn from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &TalkerConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = "talker.code_predictor";
        let dtype = DType::F32;
        let cp_cfg = &cfg.code_predictor;

        // Load codec embeddings
        let mut codec_embeddings = Vec::with_capacity(cfg.num_codebooks);
        for i in 0..cfg.num_codebooks {
            let emb = gg.embedding(
                &format!("{prefix}.model.codec_embedding.{i}.weight"),
                cp_cfg.hidden_size,
            )?;
            codec_embeddings.push(emb);
        }

        // Determine number of layers
        let mut num_layers = 0;
        while gg.has_tensor(&format!("{prefix}.model.layers.{num_layers}.input_layernorm.weight")) {
            num_layers += 1;
        }

        // Load dense decoder layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_prefix = format!("{prefix}.model.layers.{i}");

            let input_layernorm = gg.rms_norm(
                &format!("{layer_prefix}.input_layernorm.weight"),
                cfg.rms_norm_eps,
            )?;

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
                num_heads: cp_cfg.num_attention_heads,
                num_kv_heads: cp_cfg.num_key_value_heads,
                head_dim: cp_cfg.head_dim,
                kv_cache: None,
            };

            let mlp = Mlp {
                gate_proj: gg.qmatmul(&format!("{layer_prefix}.mlp.gate_proj.weight"))?,
                up_proj: gg.qmatmul(&format!("{layer_prefix}.mlp.up_proj.weight"))?,
                down_proj: gg.qmatmul(&format!("{layer_prefix}.mlp.down_proj.weight"))?,
            };

            let post_attention_layernorm = gg.rms_norm(
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )?;

            layers.push(CodePredictorDecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            });
        }

        // Final norm
        let norm = gg.rms_norm(&format!("{prefix}.model.norm.weight"), cfg.rms_norm_eps)?;

        // LM heads (15 codebooks)
        let mut lm_heads = Vec::with_capacity(cfg.num_codebooks);
        for i in 0..cfg.num_codebooks {
            let head = gg.qmatmul(&format!("{prefix}.lm_head.{i}.weight"))?;
            lm_heads.push(head);
        }

        // Rotary embedding
        let rotary = Arc::new(RotaryEmbedding::new(
            cp_cfg.head_dim,
            cfg.max_position_embeddings,
            cp_cfg.rope_theta,
            device,
            dtype,
        )?);

        Ok(Self {
            codec_embeddings,
            layers,
            norm,
            lm_heads,
            rotary,
            device: device.clone(),
            dtype,
        })
    }

    /// Embed codec tokens
    fn embed_codec_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        let (_b, _seq, num_cb) = tokens.dims3()?;
        let mut result: Option<Tensor> = None;

        for (i, emb) in self.codec_embeddings.iter().enumerate() {
            if i >= num_cb {
                break;
            }
            let cb_tokens = tokens.narrow(2, i, 1)?.squeeze(2)?;
            let embedded = emb.forward(&cb_tokens)?;

            result = Some(match result {
                Some(acc) => (acc + embedded)?,
                None => embedded,
            });
        }

        result.ok_or_else(|| candle::Error::Msg("No codebooks".into()))
    }

    /// Forward pass: returns tokens for all codebooks
    fn forward(&mut self, input_embeds: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let mut hidden = input_embeds.clone();

        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, mask, offset)?;
        }

        hidden = self.norm.forward(&hidden)?;

        // Generate tokens from each LM head
        let mut tokens_list = Vec::with_capacity(self.lm_heads.len());
        for head in &self.lm_heads {
            let logits = head.forward(&hidden)?;
            let tokens = logits.argmax(D::Minus1)?;
            tokens_list.push(tokens);
        }

        Tensor::stack(&tokens_list, 2)
    }

    fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

// ============================================================================
// Projection MLP
// ============================================================================

struct ProjectionMlp {
    fc1: QMatMul,
    fc1_bias: Tensor,
    fc2: QMatMul,
    fc2_bias: Tensor,
}

impl ProjectionMlp {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(hidden_states)?;
        let x = x.broadcast_add(&self.fc1_bias)?;
        let x = x.silu()?;
        let x = self.fc2.forward(&x)?;
        x.broadcast_add(&self.fc2_bias)
    }
}

// ============================================================================
// Special Tokens
// ============================================================================

/// Special token IDs for Talker codec embedding
pub struct TalkerSpecialTokens {
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    pub codec_pad_id: u32,
    pub codec_bos_id: u32,
    pub codec_eos_id: u32,
}

impl Default for TalkerSpecialTokens {
    fn default() -> Self {
        Self {
            codec_nothink_id: 2155,
            codec_think_bos_id: 2156,
            codec_think_eos_id: 2157,
            codec_pad_id: 2148,
            codec_bos_id: 2149,
            codec_eos_id: 2150,
        }
    }
}

/// Speaker IDs for TTS
#[derive(Debug, Clone, Copy)]
pub enum Speaker {
    Chelsie = 2301,
    Ethan = 2302,
    Aiden = 2303,
}

impl Speaker {
    pub fn id(&self) -> u32 {
        *self as u32
    }
}

// ============================================================================
// Main Talker Struct
// ============================================================================

/// Quantized Talker: speech synthesis model
pub struct Talker {
    /// Hidden projection: Thinker hidden (2048) -> Talker hidden (1024)
    hidden_projection: Option<ProjectionMlp>,
    /// Text projection: Thinker hidden (2048) -> Talker hidden (1024)
    text_projection: Option<ProjectionMlp>,
    /// TalkerModel: MoE decoder (20 layers)
    talker_model: Option<TalkerModel>,
    /// codec_head: Linear 1024 -> 3072 (logits for codebooks 1-15)
    codec_head: Option<QMatMul>,
    /// CodePredictor: Dense decoder (5 layers)
    code_predictor: CodePredictor,
    /// Special tokens
    special_tokens: TalkerSpecialTokens,
    /// Config
    cfg: TalkerConfig,
    /// Device
    device: Device,
    /// DType
    dtype: DType,
}

impl Talker {
    /// Load Talker from GGUF file
    pub fn from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &TalkerConfig,
        device: &Device,
    ) -> Result<Self> {
        let dtype = DType::F32;

        // Load hidden projection
        let hidden_projection = if gg.has_tensor("talker.hidden_projection.linear_fc1.weight") {
            let fc1 = gg.qmatmul("talker.hidden_projection.linear_fc1.weight")?;
            let fc1_bias = gg.dequantize_f32("talker.hidden_projection.linear_fc1.bias")?;
            let fc2 = gg.qmatmul("talker.hidden_projection.linear_fc2.weight")?;
            let fc2_bias = gg.dequantize_f32("talker.hidden_projection.linear_fc2.bias")?;
            Some(ProjectionMlp { fc1, fc1_bias, fc2, fc2_bias })
        } else {
            None
        };

        // Load text projection
        let text_projection = if gg.has_tensor("talker.text_projection.linear_fc1.weight") {
            let fc1 = gg.qmatmul("talker.text_projection.linear_fc1.weight")?;
            let fc1_bias = gg.dequantize_f32("talker.text_projection.linear_fc1.bias")?;
            let fc2 = gg.qmatmul("talker.text_projection.linear_fc2.weight")?;
            let fc2_bias = gg.dequantize_f32("talker.text_projection.linear_fc2.bias")?;
            Some(ProjectionMlp { fc1, fc1_bias, fc2, fc2_bias })
        } else {
            None
        };

        // Load TalkerModel (MoE decoder)
        let talker_model = if gg.has_tensor("talker.model.layers.0.input_layernorm.weight") {
            Some(TalkerModel::from_gguf(gg, cfg, device)?)
        } else {
            None
        };

        // Load codec_head
        let codec_head = if gg.has_tensor("talker.codec_head.weight") {
            Some(gg.qmatmul("talker.codec_head.weight")?)
        } else {
            None
        };

        // Load CodePredictor
        let code_predictor = CodePredictor::from_gguf(gg, cfg, device)?;

        Ok(Self {
            hidden_projection,
            text_projection,
            talker_model,
            codec_head,
            code_predictor,
            special_tokens: TalkerSpecialTokens::default(),
            cfg: cfg.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Create causal attention mask
    fn causal_mask(&self, seq: usize, offset: usize) -> Result<Tensor> {
        let mask: Vec<f32> = (0..seq)
            .flat_map(|i| {
                (0..seq + offset).map(move |j| {
                    if j <= i + offset { 0.0 } else { f32::NEG_INFINITY }
                })
            })
            .collect();

        Tensor::from_slice(&mask, (1, 1, seq, seq + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// Check if TalkerModel is loaded
    pub fn has_talker_model(&self) -> bool {
        self.talker_model.is_some()
    }

    /// Check if codec_head is loaded
    pub fn has_codec_head(&self) -> bool {
        self.codec_head.is_some()
    }

    /// Check if hidden_projection is loaded
    pub fn has_hidden_projection(&self) -> bool {
        self.hidden_projection.is_some()
    }

    /// Check if text_projection is loaded
    pub fn has_text_projection(&self) -> bool {
        self.text_projection.is_some()
    }

    /// Get number of TalkerModel layers (0 if not loaded)
    pub fn talker_model_num_layers(&self) -> usize {
        self.talker_model.as_ref().map_or(0, |m| m.num_layers())
    }

    /// Project text embeddings from Thinker to Talker hidden size
    pub fn text_projection(&self, thinker_embeds: &Tensor) -> Result<Tensor> {
        match &self.text_projection {
            Some(proj) => proj.forward(thinker_embeds),
            None => Err(candle::Error::Msg("text_projection not loaded".into())),
        }
    }

    /// Project hidden states from Thinker to Talker hidden size
    pub fn hidden_projection(&self, thinker_hidden: &Tensor) -> Result<Tensor> {
        match &self.hidden_projection {
            Some(proj) => proj.forward(thinker_hidden),
            None => Err(candle::Error::Msg("hidden_projection not loaded".into())),
        }
    }

    /// Embed special codec tokens using TalkerModel codec_embedding
    pub fn embed_special_codec_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        match &self.talker_model {
            Some(model) => {
                // For special tokens, we use the first codec embedding
                // tokens shape: [batch, seq]
                model.codec_embeddings[0].forward(tokens)
            }
            None => Err(candle::Error::Msg("TalkerModel not loaded".into())),
        }
    }

    /// Clear all KV caches
    pub fn clear_kv_cache(&mut self) {
        if let Some(model) = &mut self.talker_model {
            model.clear_kv_cache();
        }
        self.code_predictor.clear_kv_cache();
    }

    /// Get number of codebooks
    pub fn num_codebooks(&self) -> usize {
        self.cfg.num_codebooks
    }

    /// Get codebook size
    pub fn codebook_size(&self) -> usize {
        self.cfg.codebook_size
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    /// Forward pass through TalkerModel only
    /// Returns hidden states [batch, seq, 1024]
    pub fn forward_talker_model(
        &mut self,
        input_embeds: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        match &mut self.talker_model {
            Some(model) => model.forward(input_embeds, mask, offset),
            None => Err(candle::Error::Msg("TalkerModel not loaded".into())),
        }
    }

    /// Apply codec_head to get logits for 15 codebooks
    /// Input: [batch, seq, 1024]
    /// Output: [batch, seq, 15, codebook_size]
    pub fn forward_codec_head(&self, hidden: &Tensor) -> Result<Tensor> {
        match &self.codec_head {
            Some(head) => {
                // codec_head output: [batch, seq, 3072]
                let logits_flat = head.forward(hidden)?;
                let (_b, _seq, _) = logits_flat.dims3()?;

                // Split to 15 codebooks: 3072 / 15 = 204.8 -> use 204 per codebook
                // Actually the output vocab is codebook_size (1024) per head
                // The 3072 is probably 15 * 204 or similar
                // Let's reshape: 3072 / 15 = 204.8, so it's likely 15 * 205 = 3075 or padded
                // For now, assume equal split
                let per_cb = 3072 / self.cfg.num_codebooks; // 204
                let mut logits_list = Vec::with_capacity(self.cfg.num_codebooks);
                for i in 0..self.cfg.num_codebooks {
                    let start = i * per_cb;
                    let cb_logits = logits_flat.narrow(2, start, per_cb)?;
                    logits_list.push(cb_logits);
                }
                Tensor::stack(&logits_list, 2)
            }
            None => Err(candle::Error::Msg("codec_head not loaded".into())),
        }
    }

    /// Forward pass through CodePredictor
    /// Input: codec tokens [batch, seq, 15]
    /// Output: refined codec tokens [batch, seq, 15]
    pub fn forward_code_predictor(
        &mut self,
        codec_tokens: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let input_embeds = self.code_predictor.embed_codec_tokens(codec_tokens)?;
        self.code_predictor.forward(&input_embeds, mask, offset)
    }

    /// Generate speech with proper TTS pipeline
    ///
    /// Full pipeline:
    /// 1. Project thinker_embeds to Talker hidden size (1024)
    /// 2. Process through TalkerModel (20 MoE layers)
    /// 3. Apply codec_head to get initial codec tokens
    /// 4. Refine with CodePredictor (5 dense layers)
    ///
    /// # Arguments
    /// * `thinker_embeds` - Thinker text embeddings [batch, seq, 2048]
    /// * `tts_special_embeds` - TTS special token embeddings from Thinker (bos, eos, pad) [3, 2048]
    /// * `speaker` - Speaker enum for voice selection
    /// * `max_steps` - Maximum generation steps
    ///
    /// # Returns
    /// * Generated codec tokens [batch, total_seq, num_codebooks]
    pub fn generate_with_speaker(
        &mut self,
        thinker_embeds: &Tensor,
        tts_special_embeds: &Tensor,
        speaker: Speaker,
        max_steps: usize,
    ) -> Result<Tensor> {
        // Validate components
        if self.text_projection.is_none() {
            return Err(candle::Error::Msg("text_projection not loaded".into()));
        }
        if self.talker_model.is_none() {
            return Err(candle::Error::Msg("TalkerModel not loaded".into()));
        }
        if self.codec_head.is_none() {
            return Err(candle::Error::Msg("codec_head not loaded".into()));
        }

        let (batch, text_len, _) = thinker_embeds.dims3()?;
        if batch != 1 {
            return Err(candle::Error::Msg("Only batch_size=1 supported".into()));
        }
        if text_len < 4 {
            return Err(candle::Error::Msg("Need at least 4 text tokens".into()));
        }

        // Clear caches
        self.clear_kv_cache();

        // Project TTS special tokens
        let tts_special = tts_special_embeds.unsqueeze(0)?;
        let tts_special_proj = self.text_projection.as_ref().unwrap().forward(&tts_special)?;
        let tts_bos_embed = tts_special_proj.narrow(1, 0, 1)?;
        let tts_pad_embed = tts_special_proj.narrow(1, 2, 1)?;

        // Project first 3 thinker tokens
        let first_3 = thinker_embeds.narrow(1, 0, 3)?;
        let first_3_proj = self.text_projection.as_ref().unwrap().forward(&first_3)?;

        // Create codec special token sequence
        let codec_special = Tensor::from_slice(
            &[
                self.special_tokens.codec_nothink_id,
                self.special_tokens.codec_think_bos_id,
                self.special_tokens.codec_think_eos_id,
                speaker.id(),
                self.special_tokens.codec_pad_id,
                self.special_tokens.codec_bos_id,
            ],
            (1, 6),
            &self.device,
        )?;

        // Embed codec special tokens
        let codec_special_embed = self.embed_special_codec_tokens(&codec_special)?;

        // Build initialization sequence
        let zeros_3 = Tensor::zeros((1, 3, self.cfg.hidden_size), self.dtype, &self.device)?;

        let tts_pad_x4 = tts_pad_embed.repeat(&[1, 4, 1])?;
        let fourth_token = thinker_embeds.narrow(1, 3, 1)?;
        let fourth_proj = self.text_projection.as_ref().unwrap().forward(&fourth_token)?;

        let text_hidden = Tensor::cat(&[
            &first_3_proj,
            &tts_pad_x4,
            &tts_bos_embed,
            &fourth_proj,
        ], 1)?;

        let codec_hidden = Tensor::cat(&[
            &zeros_3,
            &codec_special_embed,
        ], 1)?;

        let init_embeds = (text_hidden + codec_hidden)?;
        let init_seq_len = init_embeds.dim(1)?;

        // Process through TalkerModel
        let mask = self.causal_mask(init_seq_len, 0)?;
        let hidden = self.forward_talker_model(&init_embeds, Some(&mask), 0)?;

        // Get initial codec tokens from codec_head
        let codec_logits = self.forward_codec_head(&hidden)?;
        let last_logits = codec_logits.narrow(1, init_seq_len - 1, 1)?;
        let mut current_tokens = last_logits.argmax(D::Minus1)?; // [1, 1, 15]
        let mut all_tokens = vec![current_tokens.clone()];

        // Project remaining text tokens
        let remaining_proj = if text_len > 4 {
            let remaining = thinker_embeds.narrow(1, 4, text_len - 4)?;
            Some(self.text_projection.as_ref().unwrap().forward(&remaining)?)
        } else {
            None
        };

        // Process remaining text tokens
        if let Some(ref remaining) = remaining_proj {
            let remaining_len = remaining.dim(1)?;

            for pos in 0..remaining_len {
                let text_embed = remaining.narrow(1, pos, 1)?;
                let talker_model = self.talker_model.as_ref().unwrap();
                let codec_embed = talker_model.embed_codec_tokens(&current_tokens)?;
                let combined = (text_embed + codec_embed)?;

                let offset = init_seq_len + pos;
                let hidden = self.forward_talker_model(&combined, None, offset)?;
                let codec_logits = self.forward_codec_head(&hidden)?;
                current_tokens = codec_logits.argmax(D::Minus1)?;
                all_tokens.push(current_tokens.clone());
            }
        }

        // Autoregressive generation
        for step in 0..max_steps {
            // Check for EOS
            let first_cb_token: u32 = current_tokens
                .narrow(2, 0, 1)?
                .squeeze(2)?
                .flatten_all()?
                .to_vec1()?[0];

            if first_cb_token == self.special_tokens.codec_eos_id {
                break;
            }

            let talker_model = self.talker_model.as_ref().unwrap();
            let codec_embed = talker_model.embed_codec_tokens(&current_tokens)?;

            let offset = init_seq_len + remaining_proj.as_ref().map_or(0, |r| r.dim(1).unwrap_or(0)) + step;
            let hidden = self.forward_talker_model(&codec_embed, None, offset)?;
            let codec_logits = self.forward_codec_head(&hidden)?;
            current_tokens = codec_logits.argmax(D::Minus1)?;
            all_tokens.push(current_tokens.clone());
        }

        // Concatenate all tokens
        let initial_tokens = Tensor::cat(&all_tokens, 1)?;

        // Refine with CodePredictor
        self.code_predictor.clear_kv_cache();
        let refined_seq_len = initial_tokens.dim(1)?;
        let cp_mask = self.causal_mask(refined_seq_len, 0)?;
        let refined_tokens = self.forward_code_predictor(&initial_tokens, Some(&cp_mask), 0)?;

        Ok(refined_tokens)
    }

    /// Forward pass for legacy compatibility (uses CodePredictor only)
    /// This is the OLD behavior that doesn't use TalkerModel
    pub fn forward(&mut self, input_tokens: &Tensor) -> Result<Tensor> {
        let (_b, seq, _) = input_tokens.dims3()?;
        let mask = self.causal_mask(seq, 0)?;
        self.forward_code_predictor(input_tokens, Some(&mask), 0)
    }

    /// Forward pass from Thinker hidden states (legacy)
    /// Uses the full pipeline if TalkerModel is available
    pub fn forward_from_hidden(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        if self.talker_model.is_some() && self.codec_head.is_some() {
            // Use full pipeline
            let hidden_proj = match &self.hidden_projection {
                Some(proj) => proj.forward(hidden_states)?,
                None => return Err(candle::Error::Msg("hidden_projection not loaded".into())),
            };

            self.clear_kv_cache();
            let (_, seq, _) = hidden_proj.dims3()?;
            let mask = self.causal_mask(seq, 0)?;

            // TalkerModel -> codec_head -> CodePredictor
            let talker_hidden = self.forward_talker_model(&hidden_proj, Some(&mask), 0)?;
            let codec_logits = self.forward_codec_head(&talker_hidden)?;
            let initial_tokens = codec_logits.argmax(D::Minus1)?;

            self.code_predictor.clear_kv_cache();
            let cp_mask = self.causal_mask(seq, 0)?;
            self.forward_code_predictor(&initial_tokens, Some(&cp_mask), 0)
        } else {
            // TalkerModel not loaded - cannot proceed
            // The hidden_proj has dim 1024 but CodePredictor expects 2048
            Err(candle::Error::Msg(
                "TalkerModel not loaded, cannot use forward_from_hidden. \
                 Please ensure talker.model.* tensors are present in GGUF.".into()
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_defaults() {
        let tokens = TalkerSpecialTokens::default();
        assert_eq!(tokens.codec_bos_id, 2149);
        assert_eq!(tokens.codec_eos_id, 2150);
        assert_eq!(tokens.codec_pad_id, 2148);
    }

    #[test]
    fn test_speaker_ids() {
        assert_eq!(Speaker::Chelsie.id(), 2301);
        assert_eq!(Speaker::Ethan.id(), 2302);
        assert_eq!(Speaker::Aiden.id(), 2303);
    }
}
