//! AuT (Audio-to-Token) Encoder for Qwen3-Omni
//!
//! This is a custom audio encoder (NOT Whisper-based) that converts
//! raw audio into discrete tokens for the Thinker model.
//!
//! Architecture:
//! - Conv stem: downsamples audio
//! - Transformer encoder: self-attention layers
//! - Quantizer: VQ-VAE style discrete tokenization

use super::config::AuTEncoderConfig;
use crate::models::with_tracing::{linear_no_bias, Linear, RmsNorm};
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;
use std::sync::Arc;

/// Convolutional stem for audio feature extraction
struct ConvStem {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
    norm: candle_nn::LayerNorm,
}

impl ConvStem {
    fn new(cfg: &AuTEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let conv1_cfg = candle_nn::Conv1dConfig {
            stride: 2,
            padding: 3,
            ..Default::default()
        };
        let conv2_cfg = candle_nn::Conv1dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };

        let conv1 = candle_nn::conv1d(
            cfg.n_mels,
            cfg.hidden_size / 2,
            7,
            conv1_cfg,
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(
            cfg.hidden_size / 2,
            cfg.hidden_size,
            3,
            conv2_cfg,
            vb.pp("conv2"),
        )?;
        let norm = candle_nn::layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        Ok(Self { conv1, conv2, norm })
    }
}

impl Module for ConvStem {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [batch, n_mels, frames]
        let xs = self.conv1.forward(xs)?;
        let xs = xs.gelu_erf()?;
        let xs = self.conv2.forward(&xs)?;
        let xs = xs.gelu_erf()?;

        // Transpose for layer norm: [batch, frames, hidden]
        let xs = xs.transpose(1, 2)?;
        self.norm.forward(&xs)
    }
}

/// Rotary positional embedding
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(dim: usize, max_seq_len: usize, device: &Device, dtype: DType) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..dim / 2)
            .map(|i| 1.0 / 10000_f32.powf(2.0 * i as f32 / dim as f32))
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
        let seq_len = q.dim(1)?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;

        let q_rot = apply_rotary(q, &cos, &sin)?;
        let k_rot = apply_rotary(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b, seq, n_heads, dim) = x.dims4()?;
    let x = x.reshape((b, seq, n_heads, 2, dim / 2))?;

    let x0 = x.narrow(3, 0, 1)?.squeeze(3)?;
    let x1 = x.narrow(3, 1, 1)?.squeeze(3)?;

    let cos = cos.unsqueeze(0)?.unsqueeze(2)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(2)?;

    let cos = cos.broadcast_as((b, seq, n_heads, dim / 2))?;
    let sin = sin.broadcast_as((b, seq, n_heads, dim / 2))?;

    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x1.broadcast_mul(&cos)? + x0.broadcast_mul(&sin)?)?;

    Tensor::stack(&[r0, r1], 3)?.reshape((b, seq, n_heads, dim))
}

/// Self-attention layer
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &AuTEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?,
            o_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding, offset: usize) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape for multi-head attention
        let q = q.reshape((b, seq, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq, self.num_heads, self.head_dim))?;
        let v = v.reshape((b, seq, self.num_heads, self.head_dim))?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, offset)?;

        // Transpose for attention: [b, heads, seq, dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // Transpose back and project
        let out = out.transpose(1, 2)?.reshape((b, seq, ()))?;
        self.o_proj.forward(&out)
    }
}

/// Feed-forward network
struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn new(cfg: &AuTEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let intermediate = cfg.hidden_size * 4;
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, intermediate, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, intermediate, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate, cfg.hidden_size, vb.pp("down_proj"))?,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.silu()?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Encoder layer
struct EncoderLayer {
    attn: Attention,
    mlp: MLP,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl EncoderLayer {
    fn new(cfg: &AuTEncoderConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: MLP::new(cfg, vb.pp("mlp"))?,
            ln1: RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            ln2: RmsNorm::new(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding, offset: usize) -> Result<Tensor> {
        let residual = xs;
        let xs = self.ln1.forward(xs)?;
        let xs = self.attn.forward(&xs, rotary, offset)?;
        let xs = (residual + xs)?;

        let residual = &xs;
        let xs = self.ln2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

/// Vector Quantizer for discrete tokens
#[allow(dead_code)]
struct VectorQuantizer {
    embedding: candle_nn::Embedding,
    num_embeddings: usize,
}

impl VectorQuantizer {
    fn new(cfg: &AuTEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embedding =
            candle_nn::embedding(cfg.audio_vocab_size, cfg.hidden_size, vb.pp("embedding"))?;
        Ok(Self {
            embedding,
            num_embeddings: cfg.audio_vocab_size,
        })
    }

    /// Quantize continuous embeddings to discrete tokens
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        // xs: [batch, seq, hidden]
        let (_b, _seq, _hidden) = xs.dims3()?;

        // Get all embeddings: [vocab, hidden]
        let embeddings = self.embedding.embeddings();

        // Compute distances: ||x - e||^2 = ||x||^2 - 2*x*e + ||e||^2
        let x_sq = xs.sqr()?.sum_keepdim(2)?; // [b, seq, 1]
        let e_sq = embeddings.sqr()?.sum_keepdim(1)?.t()?; // [1, vocab]

        // x @ e.T: [b, seq, vocab]
        let xe = xs.matmul(&embeddings.t()?)?;

        let xe_scaled = (xe * 2.0)?;
        let diff = x_sq.broadcast_sub(&xe_scaled)?;
        let distances = diff.broadcast_add(&e_sq)?;

        // Get nearest embedding indices
        let indices = distances.argmin(2)?; // [b, seq]

        // Get quantized embeddings
        let quantized = self.embedding.forward(&indices)?;

        Ok((quantized, indices))
    }
}

/// AuT Encoder: Audio to Tokens
#[allow(dead_code)]
pub struct AuTEncoder {
    conv_stem: ConvStem,
    layers: Vec<EncoderLayer>,
    quantizer: VectorQuantizer,
    norm: RmsNorm,
    rotary: Arc<RotaryEmbedding>,
    device: Device,
    dtype: DType,
}

impl AuTEncoder {
    pub fn new(cfg: &AuTEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let conv_stem = ConvStem::new(cfg, vb.pp("conv_stem"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        let quantizer = VectorQuantizer::new(cfg, vb.pp("quantizer"))?;
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 8192, &device, dtype)?);

        Ok(Self {
            conv_stem,
            layers,
            quantizer,
            norm,
            rotary,
            device,
            dtype,
        })
    }

    /// Encode audio mel spectrogram to discrete tokens
    ///
    /// # Arguments
    /// * `mel` - Mel spectrogram [batch, n_mels, frames]
    ///
    /// # Returns
    /// * Audio tokens [batch, seq_len]
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        // Conv stem
        let mut xs = self.conv_stem.forward(mel)?;

        // Transformer layers
        for layer in &self.layers {
            xs = layer.forward(&xs, &self.rotary, 0)?;
        }

        // Final norm
        xs = self.norm.forward(&xs)?;

        // Quantize to tokens
        let (_, tokens) = self.quantizer.forward(&xs)?;

        Ok(tokens)
    }

    /// Get continuous embeddings before quantization (for training)
    pub fn encode_continuous(&self, mel: &Tensor) -> Result<Tensor> {
        let mut xs = self.conv_stem.forward(mel)?;

        for layer in &self.layers {
            xs = layer.forward(&xs, &self.rotary, 0)?;
        }

        self.norm.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary_embedding() {
        let device = Device::Cpu;
        let rotary = RotaryEmbedding::new(64, 128, &device, DType::F32).unwrap();

        // Check shapes
        assert_eq!(rotary.cos.dims(), &[128, 64]);
        assert_eq!(rotary.sin.dims(), &[128, 64]);
    }
}
