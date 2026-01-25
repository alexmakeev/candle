//! Quantized Talker (code_predictor) for Qwen3-Omni
//!
//! Takes hidden states from the Thinker and generates multi-codebook
//! audio tokens that can be decoded by Code2Wav.
//!
//! GGUF tensor structure:
//! - talker.hidden_projection.linear_fc1.{weight,bias}: [2048, 2048] Q8_0 / F32
//! - talker.hidden_projection.linear_fc2.{weight,bias}: [1024, 2048] Q8_0 / F32
//! - talker.text_projection.linear_fc1.{weight,bias}: [2048, 2048] Q8_0 / F32
//! - talker.text_projection.linear_fc2.{weight,bias}: [1024, 2048] Q8_0 / F32
//! - talker.code_predictor.model.codec_embedding.{0-14}.weight: [2048, 1024] Q8_0
//! - talker.code_predictor.model.layers.{N}.self_attn.{q,k,v,o}_proj.weight: Q8_0
//! - talker.code_predictor.model.layers.{N}.self_attn.{q,k}_norm.weight: F32
//! - talker.code_predictor.model.layers.{N}.mlp.{gate,up,down}_proj.weight: Q8_0
//! - talker.code_predictor.model.layers.{N}.{input,post_attention}_layernorm.weight: F32
//! - talker.code_predictor.lm_head.{0-14}.weight: [2048, 1024] Q8_0

use super::config::TalkerConfig;
use super::gguf_loader::Gguf;
use crate::models::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use candle::{DType, Device, Module, Result, Tensor, D};
use std::io::{Read, Seek};
use std::sync::Arc;

/// Rotary embedding for Talker attention
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

/// Grouped Query Attention with QK normalization
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
        // Flatten [B, H, L, D] -> [B*H*L, D] for RmsNorm
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

/// MLP with gate/up/down projections (SwiGLU)
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

/// Transformer decoder layer
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn forward(
        &mut self,
        xs: &Tensor,
        rotary: &RotaryEmbedding,
        mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        // Self-attention with residual
        let residual = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward(&hidden, rotary, mask, offset)?;
        let xs = (residual + attn_out)?;

        // MLP with residual
        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        let mlp_out = self.mlp.forward(&hidden)?;
        residual + mlp_out
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Multi-codebook embedding: combines embeddings from all codebooks
struct CodecEmbedding {
    embeddings: Vec<candle_nn::Embedding>,
}

impl CodecEmbedding {
    /// Forward: sum embeddings from all codebooks
    /// Input: [batch, seq, num_codebooks] token IDs
    /// Output: [batch, seq, hidden_size]
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (_b, _seq, num_cb) = tokens.dims3()?;
        assert_eq!(num_cb, self.embeddings.len());

        let mut result: Option<Tensor> = None;

        for (i, emb) in self.embeddings.iter().enumerate() {
            // Get tokens for this codebook: [batch, seq]
            let cb_tokens = tokens.narrow(2, i, 1)?.squeeze(2)?;
            let embedded = emb.forward(&cb_tokens)?;

            result = Some(match result {
                Some(acc) => (acc + embedded)?,
                None => embedded,
            });
        }

        result.ok_or_else(|| candle::Error::Msg("No codebooks".into()))
    }
}

/// Projection MLP: projects Thinker hidden states to Talker hidden size
/// fc1: 2048 → 2048, fc2: 2048 → 1024
/// Used for both hidden_projection (multimodal) and text_projection (text)
struct ProjectionMlp {
    fc1: QMatMul,
    fc1_bias: Tensor,
    fc2: QMatMul,
    fc2_bias: Tensor,
}

impl ProjectionMlp {
    /// Forward: project from Thinker hidden (2048) to Talker hidden (1024)
    /// Uses SiLU activation between fc1 and fc2
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // fc1: [B, S, 2048] → [B, S, 2048]
        let x = self.fc1.forward(hidden_states)?;
        let x = x.broadcast_add(&self.fc1_bias)?;
        let x = x.silu()?;

        // fc2: [B, S, 2048] → [B, S, 1024]
        let x = self.fc2.forward(&x)?;
        x.broadcast_add(&self.fc2_bias)
    }
}

/// Special token IDs for Talker codec embedding
/// From HuggingFace Qwen3-Omni config
pub struct TalkerSpecialTokens {
    /// No-think token for non-thinking mode
    pub codec_nothink_id: u32,
    /// Think begin-of-sequence token
    pub codec_think_bos_id: u32,
    /// Think end-of-sequence token
    pub codec_think_eos_id: u32,
    /// Padding token for codec
    pub codec_pad_id: u32,
    /// Begin-of-sequence token for codec
    pub codec_bos_id: u32,
    /// End-of-sequence token for codec
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

/// Multi-head LM output: one head per codebook
struct LmHeads {
    heads: Vec<QMatMul>,
}

impl LmHeads {
    /// Forward: compute logits for each codebook
    /// Input: [batch, seq, hidden_size]
    /// Output: [batch, seq, num_codebooks, codebook_size]
    fn forward(&self, hidden: &Tensor) -> Result<Tensor> {
        let mut logits_list = Vec::with_capacity(self.heads.len());

        for head in &self.heads {
            let logits = head.forward(hidden)?; // [batch, seq, codebook_size]
            logits_list.push(logits);
        }

        // Stack along dim 2: [batch, seq, num_codebooks, codebook_size]
        Tensor::stack(&logits_list, 2)
    }

    /// Generate tokens: argmax over logits
    /// Returns: [batch, seq, num_codebooks]
    fn generate(&self, hidden: &Tensor) -> Result<Tensor> {
        let mut tokens_list = Vec::with_capacity(self.heads.len());

        for head in &self.heads {
            let logits = head.forward(hidden)?; // [batch, seq, codebook_size]
            let tokens = logits.argmax(D::Minus1)?; // [batch, seq]
            tokens_list.push(tokens);
        }

        Tensor::stack(&tokens_list, 2)
    }
}

/// Quantized Talker: speech synthesis model
pub struct Talker {
    /// Hidden projection: Thinker hidden (2048) → Talker hidden (1024) for multimodal
    hidden_projection: Option<ProjectionMlp>,
    /// Text projection: Thinker hidden (2048) → Talker hidden (1024) for text
    text_projection: Option<ProjectionMlp>,
    /// Special codec embedding from talker.model (vocab_size=3072, for special tokens 2148-2302)
    special_codec_embedding: Option<candle_nn::Embedding>,
    /// Codec embeddings (15 codebooks) from code_predictor
    codec_embedding: CodecEmbedding,
    /// Transformer layers
    layers: Vec<DecoderLayer>,
    /// Final norm
    norm: RmsNorm,
    /// LM heads (15 codebooks)
    lm_heads: LmHeads,
    /// Rotary embedding
    rotary: Arc<RotaryEmbedding>,
    /// Special tokens for codec
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
        let prefix = "talker.code_predictor";
        let dtype = DType::F32;

        // Load hidden projection (Thinker 2048 → Talker 1024) for multimodal
        let hidden_projection = if gg.has_tensor("talker.hidden_projection.linear_fc1.weight") {
            let fc1 = gg.qmatmul("talker.hidden_projection.linear_fc1.weight")?;
            let fc1_bias = gg.dequantize_f32("talker.hidden_projection.linear_fc1.bias")?;
            let fc2 = gg.qmatmul("talker.hidden_projection.linear_fc2.weight")?;
            let fc2_bias = gg.dequantize_f32("talker.hidden_projection.linear_fc2.bias")?;
            Some(ProjectionMlp {
                fc1,
                fc1_bias,
                fc2,
                fc2_bias,
            })
        } else {
            None
        };

        // Load text projection (Thinker 2048 → Talker 1024) for text tokens
        let text_projection = if gg.has_tensor("talker.text_projection.linear_fc1.weight") {
            let fc1 = gg.qmatmul("talker.text_projection.linear_fc1.weight")?;
            let fc1_bias = gg.dequantize_f32("talker.text_projection.linear_fc1.bias")?;
            let fc2 = gg.qmatmul("talker.text_projection.linear_fc2.weight")?;
            let fc2_bias = gg.dequantize_f32("talker.text_projection.linear_fc2.bias")?;
            Some(ProjectionMlp {
                fc1,
                fc1_bias,
                fc2,
                fc2_bias,
            })
        } else {
            None
        };

        // Load special codec embedding from talker.model (vocab_size=3072)
        // This is for special tokens like speaker_id, codec_bos, codec_pad, etc.
        let special_codec_embedding = if gg.has_tensor("talker.model.codec_embedding.weight") {
            Some(gg.embedding("talker.model.codec_embedding.weight", cfg.hidden_size)?)
        } else {
            None
        };

        // Load codec embeddings (15 codebooks)
        let mut embeddings = Vec::with_capacity(cfg.num_codebooks);
        for i in 0..cfg.num_codebooks {
            let emb = gg.embedding(
                &format!("{prefix}.model.codec_embedding.{i}.weight"),
                cfg.hidden_size,
            )?;
            embeddings.push(emb);
        }
        let codec_embedding = CodecEmbedding { embeddings };

        // Determine number of layers by probing GGUF
        let mut num_layers = 0;
        while gg.has_tensor(&format!("{prefix}.model.layers.{num_layers}.input_layernorm.weight")) {
            num_layers += 1;
        }

        // Load transformer layers
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
                num_heads: cfg.num_attention_heads,
                num_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
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

            layers.push(DecoderLayer {
                self_attn,
                mlp,
                input_layernorm,
                post_attention_layernorm,
            });
        }

        // Final norm
        let norm = gg.rms_norm(&format!("{prefix}.model.norm.weight"), cfg.rms_norm_eps)?;

        // Load LM heads (15 codebooks)
        let mut heads = Vec::with_capacity(cfg.num_codebooks);
        for i in 0..cfg.num_codebooks {
            let head = gg.qmatmul(&format!("{prefix}.lm_head.{i}.weight"))?;
            heads.push(head);
        }
        let lm_heads = LmHeads { heads };

        // Create rotary embedding
        let rotary = Arc::new(RotaryEmbedding::new(
            cfg.head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            device,
            dtype,
        )?);

        Ok(Self {
            hidden_projection,
            text_projection,
            special_codec_embedding,
            codec_embedding,
            layers,
            norm,
            lm_heads,
            rotary,
            special_tokens: TalkerSpecialTokens::default(),
            cfg: cfg.clone(),
            device: device.clone(),
            dtype,
        })
    }

    /// Forward pass: generate next codec tokens
    ///
    /// # Arguments
    /// * `input_tokens` - Previous codec tokens [batch, seq, num_codebooks]
    ///
    /// # Returns
    /// * Logits for all codebooks [batch, seq, num_codebooks, codebook_size]
    pub fn forward_logits(&mut self, input_tokens: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = input_tokens.dims3()?;

        // Embed tokens
        let mut hidden = self.codec_embedding.forward(input_tokens)?;

        // Get current sequence position for KV cache
        let offset = self.layers.first()
            .and_then(|l| l.self_attn.kv_cache.as_ref())
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0);

        // Create causal mask
        let mask = self.causal_mask(b, seq, offset)?;

        // Process through transformer layers
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), offset)?;
        }

        // Final norm
        hidden = self.norm.forward(&hidden)?;

        // LM heads
        self.lm_heads.forward(&hidden)
    }

    /// Forward pass: generate codec tokens (greedy)
    ///
    /// # Arguments
    /// * `input_tokens` - Previous codec tokens [batch, seq, num_codebooks]
    ///
    /// # Returns
    /// * Generated tokens [batch, seq, num_codebooks]
    pub fn forward(&mut self, input_tokens: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = input_tokens.dims3()?;

        // Embed tokens
        let mut hidden = self.codec_embedding.forward(input_tokens)?;

        // Get current sequence position for KV cache
        let offset = self.layers.first()
            .and_then(|l| l.self_attn.kv_cache.as_ref())
            .map(|(k, _)| k.dim(2).unwrap_or(0))
            .unwrap_or(0);

        // Create causal mask
        let mask = self.causal_mask(b, seq, offset)?;

        // Process through transformer layers
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), offset)?;
        }

        // Final norm
        hidden = self.norm.forward(&hidden)?;

        // Generate tokens
        self.lm_heads.generate(&hidden)
    }

    /// Forward pass from Thinker hidden states
    ///
    /// # Arguments
    /// * `hidden_states` - Hidden states from Thinker [batch, seq, 2048]
    ///
    /// # Returns
    /// * Generated tokens [batch, seq, num_codebooks]
    ///
    /// # Note
    /// Requires hidden_projection to be loaded from GGUF.
    /// Clears KV cache before processing.
    pub fn forward_from_hidden(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        // Project from Thinker hidden (2048) to Talker hidden (1024)
        let hidden_proj = match &self.hidden_projection {
            Some(proj) => proj.forward(hidden_states)?,
            None => {
                return Err(candle::Error::Msg(
                    "hidden_projection not loaded, cannot use forward_from_hidden".into(),
                ))
            }
        };

        let (b, seq, _) = hidden_proj.dims3()?;

        // Clear KV cache for fresh generation
        self.clear_kv_cache();

        // Create causal mask
        let mask = self.causal_mask(b, seq, 0)?;

        // Process through transformer layers
        let mut hidden = hidden_proj;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), 0)?;
        }

        // Final norm
        hidden = self.norm.forward(&hidden)?;

        // Generate tokens
        self.lm_heads.generate(&hidden)
    }

    /// Forward pass from Thinker hidden states, returns logits
    ///
    /// # Arguments
    /// * `hidden_states` - Hidden states from Thinker [batch, seq, 2048]
    ///
    /// # Returns
    /// * Logits for all codebooks [batch, seq, num_codebooks, codebook_size]
    pub fn forward_from_hidden_logits(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        // Project from Thinker hidden (2048) to Talker hidden (1024)
        let hidden_proj = match &self.hidden_projection {
            Some(proj) => proj.forward(hidden_states)?,
            None => {
                return Err(candle::Error::Msg(
                    "hidden_projection not loaded, cannot use forward_from_hidden_logits".into(),
                ))
            }
        };

        let (b, seq, _) = hidden_proj.dims3()?;

        // Clear KV cache for fresh generation
        self.clear_kv_cache();

        // Create causal mask
        let mask = self.causal_mask(b, seq, 0)?;

        // Process through transformer layers
        let mut hidden = hidden_proj;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), 0)?;
        }

        // Final norm
        hidden = self.norm.forward(&hidden)?;

        // Return logits
        self.lm_heads.forward(&hidden)
    }

    /// Check if hidden_projection is loaded
    pub fn has_hidden_projection(&self) -> bool {
        self.hidden_projection.is_some()
    }

    /// Check if text_projection is loaded
    pub fn has_text_projection(&self) -> bool {
        self.text_projection.is_some()
    }

    /// Project text embeddings from Thinker to Talker hidden size
    pub fn text_projection(&self, thinker_embeds: &Tensor) -> Result<Tensor> {
        match &self.text_projection {
            Some(proj) => proj.forward(thinker_embeds),
            None => Err(candle::Error::Msg(
                "text_projection not loaded".into(),
            )),
        }
    }

    /// Get codec embedding for special tokens (speaker, bos, pad, etc.)
    /// Uses talker.model.codec_embedding (vocab_size=3072)
    /// Input: tensor of token IDs [batch, seq]
    /// Output: embeddings [batch, seq, hidden_size]
    pub fn embed_special_codec_tokens(&self, tokens: &Tensor) -> Result<Tensor> {
        match &self.special_codec_embedding {
            Some(emb) => emb.forward(tokens),
            None => Err(candle::Error::Msg(
                "special_codec_embedding not loaded".into(),
            )),
        }
    }

    /// Check if special_codec_embedding is loaded
    pub fn has_special_codec_embedding(&self) -> bool {
        self.special_codec_embedding.is_some()
    }

    /// Generate speech with proper initialization sequence
    ///
    /// This method implements the correct Talker initialization as per HuggingFace:
    /// 1. First 3 positions: text_projection(thinker[:3]) + zeros (no codec embedding)
    /// 2. Next 3 positions: tts_pad_embed (x3) + codec_embed([speaker, pad, bos])
    /// 3. Next 1 position: tts_bos_embed + codec_embed for bos
    /// 4. First text position: text_projection(thinker[3:4]) + ...
    /// 5. Autoregressive generation for remaining text
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
        if self.text_projection.is_none() {
            return Err(candle::Error::Msg(
                "text_projection not loaded, cannot use generate_with_speaker".into(),
            ));
        }
        if self.special_codec_embedding.is_none() {
            return Err(candle::Error::Msg(
                "special_codec_embedding not loaded, cannot use generate_with_speaker".into(),
            ));
        }

        let (batch, text_len, _) = thinker_embeds.dims3()?;
        if batch != 1 {
            return Err(candle::Error::Msg(
                "generate_with_speaker only supports batch_size=1".into(),
            ));
        }
        if text_len < 4 {
            return Err(candle::Error::Msg(
                "Need at least 4 text tokens for proper initialization".into(),
            ));
        }

        // Clear KV cache for fresh generation
        self.clear_kv_cache();

        // Helper closure for text projection
        let text_proj_forward = |proj: &ProjectionMlp, x: &Tensor| -> Result<Tensor> {
            proj.forward(x)
        };

        // Project TTS special tokens (tts_bos, tts_eos, tts_pad) through text_projection
        // tts_special_embeds shape: [3, 2048] -> [1, 3, 2048]
        let tts_special = tts_special_embeds.unsqueeze(0)?;
        let tts_special_proj = text_proj_forward(
            self.text_projection.as_ref().unwrap(),
            &tts_special
        )?; // [1, 3, 1024]
        let tts_bos_embed = tts_special_proj.narrow(1, 0, 1)?; // [1, 1, 1024]
        let _tts_eos_embed = tts_special_proj.narrow(1, 1, 1)?; // [1, 1, 1024]
        let tts_pad_embed = tts_special_proj.narrow(1, 2, 1)?; // [1, 1, 1024]

        // Project first 3 thinker tokens (typically: im_start, system tokens)
        let first_3 = thinker_embeds.narrow(1, 0, 3)?;
        let first_3_proj = text_proj_forward(
            self.text_projection.as_ref().unwrap(),
            &first_3
        )?; // [1, 3, 1024]

        // Create codec special token sequence: [nothink, think_bos, think_eos, speaker, pad, bos]
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

        // Embed codec special tokens using special_codec_embedding (vocab_size=3072)
        let codec_special_embed = self.embed_special_codec_tokens(&codec_special)?; // [1, 6, 1024]

        // Build the initialization sequence (9 positions total):
        // Text part: [first_3] + [tts_pad x 4] + [tts_bos] + [fourth_token]
        // Codec part: [zeros(3)] + [codec_embed(6 special tokens)]

        let zeros_3 = Tensor::zeros((1, 3, self.cfg.hidden_size), self.dtype, &self.device)?;

        // Build text part
        let tts_pad_x4 = tts_pad_embed.repeat(&[1, 4, 1])?; // [1, 4, 1024]
        let fourth_token = thinker_embeds.narrow(1, 3, 1)?;
        let fourth_proj = text_proj_forward(
            self.text_projection.as_ref().unwrap(),
            &fourth_token
        )?; // [1, 1, 1024]

        let text_hidden = Tensor::cat(&[
            &first_3_proj,  // [1, 3, 1024]
            &tts_pad_x4,    // [1, 4, 1024]
            &tts_bos_embed, // [1, 1, 1024]
            &fourth_proj,   // [1, 1, 1024]
        ], 1)?; // [1, 9, 1024]

        // Build codec part: zeros(3) + codec_embed(6 special tokens)
        let codec_hidden = Tensor::cat(&[
            &zeros_3,           // [1, 3, 1024]
            &codec_special_embed, // [1, 6, 1024]
        ], 1)?; // [1, 9, 1024]

        // Combine
        let init_embeds = (text_hidden + codec_hidden)?; // [1, 9, 1024]

        // Process initial sequence through transformer
        let init_seq_len = 9;
        let mask = self.causal_mask(1, init_seq_len, 0)?;

        let mut hidden = init_embeds;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, &self.rotary, Some(&mask), 0)?;
        }
        hidden = self.norm.forward(&hidden)?;

        // Generate initial tokens from last position
        let last_hidden = hidden.narrow(1, init_seq_len - 1, 1)?;
        let mut current_tokens = self.lm_heads.generate(&last_hidden)?; // [1, 1, 15]
        let mut all_tokens = vec![current_tokens.clone()];

        // Project remaining text tokens if any
        let remaining_proj = if text_len > 4 {
            let remaining = thinker_embeds.narrow(1, 4, text_len - 4)?;
            Some(text_proj_forward(
                self.text_projection.as_ref().unwrap(),
                &remaining
            )?)
        } else {
            None
        };

        // Process remaining text tokens
        if let Some(ref remaining) = remaining_proj {
            let remaining_len = remaining.dim(1)?;

            for pos in 0..remaining_len {
                let text_embed = remaining.narrow(1, pos, 1)?; // [1, 1, 1024]
                let codec_embed = self.codec_embedding.forward(&current_tokens)?; // [1, 1, 1024]
                let combined = (text_embed + codec_embed)?;

                // Get current offset from KV cache
                let offset = self.layers.first()
                    .and_then(|l| l.self_attn.kv_cache.as_ref())
                    .map(|(k, _)| k.dim(2).unwrap_or(0))
                    .unwrap_or(0);

                // No mask needed for single token with KV cache
                let mut h = combined;
                for layer in &mut self.layers {
                    h = layer.forward(&h, &self.rotary, None, offset)?;
                }
                h = self.norm.forward(&h)?;

                current_tokens = self.lm_heads.generate(&h)?;
                all_tokens.push(current_tokens.clone());
            }
        }

        // Autoregressive generation loop
        for _step in 0..max_steps {
            // Check for EOS (codec_eos_id in first codebook)
            let first_cb_token: u32 = current_tokens
                .narrow(2, 0, 1)?
                .squeeze(2)?
                .flatten_all()?
                .to_vec1()?[0];

            if first_cb_token == self.special_tokens.codec_eos_id {
                break;
            }

            // Embed current tokens
            let codec_embed = self.codec_embedding.forward(&current_tokens)?;

            // Get offset
            let offset = self.layers.first()
                .and_then(|l| l.self_attn.kv_cache.as_ref())
                .map(|(k, _)| k.dim(2).unwrap_or(0))
                .unwrap_or(0);

            // Forward through layers
            let mut h = codec_embed;
            for layer in &mut self.layers {
                h = layer.forward(&h, &self.rotary, None, offset)?;
            }
            h = self.norm.forward(&h)?;

            // Generate next tokens
            current_tokens = self.lm_heads.generate(&h)?;
            all_tokens.push(current_tokens.clone());
        }

        // Concatenate all generated tokens
        Tensor::cat(&all_tokens, 1)
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

    /// Clear KV cache
    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = TalkerConfig::default();
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_codebooks, 15);
        assert_eq!(cfg.codebook_size, 2048);
    }
}
