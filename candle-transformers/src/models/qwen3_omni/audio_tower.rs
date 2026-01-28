//! Audio Tower for Qwen3-Omni
//!
//! Integrated audio encoder that is part of the Thinker model.
//! Weights at: thinker.audio_tower.*
//!
//! Architecture:
//! - Conv stem: 3 conv2d layers (480 channels)
//! - 32 Transformer layers (hidden=1280)
//! - Layer norm (ln_post)
//! - Projection layers (proj1, proj2) to Thinker hidden size

use super::config::AudioTowerConfig;
use crate::models::with_tracing::{linear_no_bias, Linear};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use std::sync::Arc;

/// Convolutional stem for audio feature extraction
/// Weights: conv2d1, conv2d2, conv2d3
struct ConvStem {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    conv3: candle_nn::Conv2d,
}

impl ConvStem {
    fn new(cfg: &AudioTowerConfig, vb: VarBuilder) -> Result<Self> {
        // conv2d1: [480, 1, 3, 3] - input is mono spectrogram
        let conv1_cfg = candle_nn::Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::conv2d(1, cfg.conv_channels, 3, conv1_cfg, vb.pp("conv2d1"))?;

        // conv2d2: [480, 480, 3, 3]
        let conv2_cfg = candle_nn::Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv2 = candle_nn::conv2d(
            cfg.conv_channels,
            cfg.conv_channels,
            3,
            conv2_cfg,
            vb.pp("conv2d2"),
        )?;

        // conv2d3: [480, 480, 3, 3]
        let conv3 = candle_nn::conv2d(
            cfg.conv_channels,
            cfg.conv_channels,
            3,
            conv2_cfg,
            vb.pp("conv2d3"),
        )?;

        Ok(Self { conv1, conv2, conv3 })
    }
}

impl Module for ConvStem {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [batch, 1, freq, time]
        let xs = self.conv1.forward(xs)?;
        let xs = xs.gelu_erf()?;
        let xs = self.conv2.forward(&xs)?;
        let xs = xs.gelu_erf()?;
        let xs = self.conv3.forward(&xs)?;
        xs.gelu_erf()
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

/// Self-attention layer
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(cfg: &AudioTowerConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        Ok(Self {
            q_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?,
            v_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?,
            out_proj: linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))?,
            num_heads: cfg.num_attention_heads,
            head_dim,
        })
    }

    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding, offset: usize) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape for multi-head attention [b, heads, seq, dim]
        let q = q
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = rotary.apply(&q, &k, offset)?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // Transpose back and project
        let out = out.transpose(1, 2)?.reshape((b, seq, ()))?;
        self.out_proj.forward(&out)
    }
}

/// Feed-forward network (MLP)
/// Weights: fc1, fc2
struct MLP {
    fc1: Linear,
    fc2: Linear,
}

impl MLP {
    fn new(cfg: &AudioTowerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?,
            fc2: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?.gelu_erf()?;
        self.fc2.forward(&xs)
    }
}

/// Encoder layer
/// Weights: self_attn.*, mlp.*, layer_norm1, layer_norm2
struct EncoderLayer {
    attn: Attention,
    mlp: MLP,
    ln1: candle_nn::LayerNorm,
    ln2: candle_nn::LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &AudioTowerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: MLP::new(cfg, vb.pp("mlp"))?,
            ln1: candle_nn::layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("layer_norm1"))?,
            ln2: candle_nn::layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("layer_norm2"))?,
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

/// Audio Tower: Encodes audio for the Thinker model
pub struct AudioTower {
    conv_stem: ConvStem,
    embed_proj: Linear,
    layers: Vec<EncoderLayer>,
    ln_post: candle_nn::LayerNorm,
    proj1: Linear,
    proj2: Linear,
    rotary: Arc<RotaryEmbedding>,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl AudioTower {
    pub fn new(cfg: &AudioTowerConfig, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let conv_stem = ConvStem::new(cfg, vb.clone())?;

        // Project conv output to hidden size
        // Conv output: [batch, conv_channels, freq', time']
        // Need to flatten freq' dim and project to hidden_size
        let embed_proj = linear_no_bias(cfg.conv_channels, cfg.hidden_size, vb.pp("embed_proj"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(EncoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        let ln_post = candle_nn::layer_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("ln_post"))?;

        // Projection layers to Thinker hidden size
        let proj1 = linear_no_bias(cfg.hidden_size, cfg.hidden_size, vb.pp("proj1"))?;
        let proj2 = linear_no_bias(cfg.hidden_size, cfg.output_size, vb.pp("proj2"))?;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 8192, &device, dtype)?);

        Ok(Self {
            conv_stem,
            embed_proj,
            layers,
            ln_post,
            proj1,
            proj2,
            rotary,
            device,
            dtype,
        })
    }

    /// Encode audio features to embeddings for Thinker
    ///
    /// # Arguments
    /// * `audio_features` - Input audio features [batch, 1, freq, time]
    ///
    /// # Returns
    /// * Audio embeddings [batch, seq_len, thinker_hidden_size]
    pub fn forward(&self, audio_features: &Tensor) -> Result<Tensor> {
        // Conv stem: [batch, 1, freq, time] -> [batch, conv_channels, freq', time']
        let xs = self.conv_stem.forward(audio_features)?;

        // Reshape: [batch, channels, freq', time'] -> [batch, time', channels * freq']
        let (b, c, f, t) = xs.dims4()?;
        let xs = xs.permute((0, 3, 1, 2))?.reshape((b, t, c * f))?;

        // Project to hidden size
        let mut xs = self.embed_proj.forward(&xs)?;

        // Transformer layers
        for layer in &self.layers {
            xs = layer.forward(&xs, &self.rotary, 0)?;
        }

        // Final layer norm
        xs = self.ln_post.forward(&xs)?;

        // Projection to Thinker hidden size
        let xs = self.proj1.forward(&xs)?;
        let xs = xs.gelu_erf()?;
        self.proj2.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary_embedding() {
        let device = Device::Cpu;
        let rotary = RotaryEmbedding::new(80, 128, &device, DType::F32).unwrap();

        assert_eq!(rotary.cos.dims(), &[128, 80]);
        assert_eq!(rotary.sin.dims(), &[128, 80]);
    }
}
