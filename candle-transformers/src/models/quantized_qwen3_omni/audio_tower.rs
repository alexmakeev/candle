//! Quantized Audio Tower for Qwen3-Omni
//!
//! Audio encoder that is part of the Thinker model.
//!
//! GGUF tensor structure:
//! - thinker.audio_tower.conv2d{1,2,3}.{weight,bias}: Conv2d layers [480, 1/480, 3, 3]
//! - thinker.audio_tower.conv_out.weight: [1280, 7680] Q8_0
//! - thinker.audio_tower.layers.{N}.self_attn.{q,k,v,out}_proj.{weight,bias}: Q8_0/F32
//! - thinker.audio_tower.layers.{N}.fc1, fc2: Q8_0
//! - thinker.audio_tower.layers.{N}.{self_attn_layer_norm,final_layer_norm}: F32
//! - thinker.audio_tower.ln_post.{weight,bias}: F32
//! - thinker.audio_tower.proj1.{weight,bias}: [1280, 1280] Q8_0/F32
//! - thinker.audio_tower.proj2.{weight,bias}: [2048, 1280] Q8_0/F32

use super::config::AudioTowerConfig;
use super::gguf_loader::Gguf;
use crate::models::with_tracing::QMatMul;
use candle::{DType, Device, Module, Result, Tensor, D};
use std::io::{Read, Seek};
use std::sync::Arc;

/// Convolutional stem for audio feature extraction
/// GGUF: conv2d{1,2,3}.{weight,bias}
struct ConvStem {
    conv1: candle_nn::Conv2d,
    conv2: candle_nn::Conv2d,
    conv3: candle_nn::Conv2d,
}

impl ConvStem {
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

/// Rotary positional embedding for audio transformer
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

/// Quantized self-attention with bias
/// GGUF: self_attn.{q,k,v,out}_proj.{weight,bias}
struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    out_proj: QMatMul,
    q_bias: Tensor,
    k_bias: Tensor,
    v_bias: Tensor,
    out_bias: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding, offset: usize) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?.broadcast_add(&self.q_bias)?;
        let k = self.k_proj.forward(xs)?.broadcast_add(&self.k_bias)?;
        let v = self.v_proj.forward(xs)?.broadcast_add(&self.v_bias)?;

        // Reshape for multi-head attention [b, seq, heads, dim] -> [b, heads, seq, dim]
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
        let out = self.out_proj.forward(&out)?;
        out.broadcast_add(&self.out_bias)
    }
}

/// MLP with GELU activation
/// GGUF: fc1, fc2
struct Mlp {
    fc1: QMatMul,
    fc1_bias: Tensor,
    fc2: QMatMul,
    fc2_bias: Tensor,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        let xs = xs.broadcast_add(&self.fc1_bias)?;
        let xs = xs.gelu_erf()?;
        let xs = self.fc2.forward(&xs)?;
        xs.broadcast_add(&self.fc2_bias)
    }
}

/// Audio encoder layer (Whisper-style)
/// GGUF: layers.{N}.self_attn.*, fc1, fc2, self_attn_layer_norm, final_layer_norm
struct EncoderLayer {
    attn: Attention,
    mlp: Mlp,
    ln1: candle_nn::LayerNorm,
    ln2: candle_nn::LayerNorm,
}

impl EncoderLayer {
    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding, offset: usize) -> Result<Tensor> {
        // Pre-norm attention
        let residual = xs;
        let xs = self.ln1.forward(xs)?;
        let xs = self.attn.forward(&xs, rotary, offset)?;
        let xs = (residual + xs)?;

        // Pre-norm MLP
        let residual = &xs;
        let xs = self.ln2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        residual + xs
    }
}

/// Quantized Audio Tower
/// Encodes mel spectrogram features to embeddings for Thinker
pub struct AudioTower {
    conv_stem: ConvStem,
    conv_out: QMatMul,
    layers: Vec<EncoderLayer>,
    ln_post: candle_nn::LayerNorm,
    proj1: QMatMul,
    proj1_bias: Tensor,
    proj2: QMatMul,
    proj2_bias: Tensor,
    rotary: Arc<RotaryEmbedding>,
    #[allow(dead_code)]
    cfg: AudioTowerConfig,
    #[allow(dead_code)]
    device: Device,
}

impl AudioTower {
    /// Load from GGUF file
    pub fn from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &AudioTowerConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = "thinker.audio_tower";

        // Conv stem: 3 conv2d layers
        // conv2d1: [480, 1, 3, 3] stride=1
        let conv1_weight = gg.dequantize(&format!("{prefix}.conv2d1.weight"))?;
        let conv1_bias = gg.dequantize(&format!("{prefix}.conv2d1.bias"))?;
        let conv1_cfg = candle_nn::Conv2dConfig {
            stride: 1,
            padding: 1,
            ..Default::default()
        };
        let conv1 = candle_nn::Conv2d::new(conv1_weight, Some(conv1_bias), conv1_cfg);

        // conv2d2: [480, 480, 3, 3] stride=2
        let conv2_weight = gg.dequantize(&format!("{prefix}.conv2d2.weight"))?;
        let conv2_bias = gg.dequantize(&format!("{prefix}.conv2d2.bias"))?;
        let conv2_cfg = candle_nn::Conv2dConfig {
            stride: 2,
            padding: 1,
            ..Default::default()
        };
        let conv2 = candle_nn::Conv2d::new(conv2_weight, Some(conv2_bias), conv2_cfg);

        // conv2d3: [480, 480, 3, 3] stride=2
        let conv3_weight = gg.dequantize(&format!("{prefix}.conv2d3.weight"))?;
        let conv3_bias = gg.dequantize(&format!("{prefix}.conv2d3.bias"))?;
        let conv3 = candle_nn::Conv2d::new(conv3_weight, Some(conv3_bias), conv2_cfg);

        let conv_stem = ConvStem { conv1, conv2, conv3 };

        // conv_out: [1280, 7680] projects flattened conv output to hidden_size
        // 7680 = 480 * 16 (conv_channels * reduced_freq_dim)
        let conv_out = gg.qmatmul(&format!("{prefix}.conv_out.weight"))?;

        // Transformer layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_prefix = format!("{prefix}.layers.{i}");

            // Attention
            let attn = Attention {
                q_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.q_proj.weight"))?,
                k_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.k_proj.weight"))?,
                v_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.v_proj.weight"))?,
                out_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.out_proj.weight"))?,
                q_bias: gg.dequantize_f32(&format!("{layer_prefix}.self_attn.q_proj.bias"))?,
                k_bias: gg.dequantize_f32(&format!("{layer_prefix}.self_attn.k_proj.bias"))?,
                v_bias: gg.dequantize_f32(&format!("{layer_prefix}.self_attn.v_proj.bias"))?,
                out_bias: gg.dequantize_f32(&format!("{layer_prefix}.self_attn.out_proj.bias"))?,
                num_heads: cfg.num_attention_heads,
                head_dim: cfg.hidden_size / cfg.num_attention_heads,
            };

            // MLP
            let mlp = Mlp {
                fc1: gg.qmatmul(&format!("{layer_prefix}.fc1.weight"))?,
                fc1_bias: gg.dequantize_f32(&format!("{layer_prefix}.fc1.bias"))?,
                fc2: gg.qmatmul(&format!("{layer_prefix}.fc2.weight"))?,
                fc2_bias: gg.dequantize_f32(&format!("{layer_prefix}.fc2.bias"))?,
            };

            // Layer norms
            let ln1 = gg.layer_norm(
                &format!("{layer_prefix}.self_attn_layer_norm.weight"),
                &format!("{layer_prefix}.self_attn_layer_norm.bias"),
                cfg.rms_norm_eps,
            )?;
            let ln2 = gg.layer_norm(
                &format!("{layer_prefix}.final_layer_norm.weight"),
                &format!("{layer_prefix}.final_layer_norm.bias"),
                cfg.rms_norm_eps,
            )?;

            layers.push(EncoderLayer { attn, mlp, ln1, ln2 });
        }

        // Post layer norm
        let ln_post = gg.layer_norm(
            &format!("{prefix}.ln_post.weight"),
            &format!("{prefix}.ln_post.bias"),
            cfg.rms_norm_eps,
        )?;

        // Projection layers to Thinker hidden size
        let proj1 = gg.qmatmul(&format!("{prefix}.proj1.weight"))?;
        let proj1_bias = gg.dequantize_f32(&format!("{prefix}.proj1.bias"))?;
        let proj2 = gg.qmatmul(&format!("{prefix}.proj2.weight"))?;
        let proj2_bias = gg.dequantize_f32(&format!("{prefix}.proj2.bias"))?;

        // Rotary embedding
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 8192, device, DType::F32)?);

        Ok(Self {
            conv_stem,
            conv_out,
            layers,
            ln_post,
            proj1,
            proj1_bias,
            proj2,
            proj2_bias,
            rotary,
            cfg: cfg.clone(),
            device: device.clone(),
        })
    }

    /// Encode audio features to embeddings for Thinker
    ///
    /// # Arguments
    /// * `audio_features` - Mel spectrogram [batch, 1, freq=128, time]
    ///
    /// # Returns
    /// * Audio embeddings [batch, seq, output_size=2048]
    pub fn forward(&self, audio_features: &Tensor) -> Result<Tensor> {
        // Conv stem: [batch, 1, freq, time] -> [batch, 480, freq', time']
        // After two stride=2 convs: freq' = freq/4, time' = time/4
        let xs = self.conv_stem.forward(audio_features)?;

        // Reshape: [batch, channels, freq', time'] -> [batch, time', channels * freq']
        let (b, c, f, t) = xs.dims4()?;
        let xs = xs.permute((0, 3, 1, 2))?.reshape((b, t, c * f))?;

        // Project to hidden_size via conv_out
        // [batch, time', 7680] -> [batch, time', 1280]
        let mut xs = self.conv_out.forward(&xs)?;

        // Transformer layers
        for layer in &self.layers {
            xs = layer.forward(&xs, &self.rotary, 0)?;
        }

        // Post layer norm
        xs = self.ln_post.forward(&xs)?;

        // Projection to Thinker hidden size (1280 -> 2048)
        let xs = self.proj1.forward(&xs)?;
        let xs = xs.broadcast_add(&self.proj1_bias)?;
        let xs = xs.gelu_erf()?;
        let xs = self.proj2.forward(&xs)?;
        xs.broadcast_add(&self.proj2_bias)
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
