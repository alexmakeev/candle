//! Quantized Code2Wav: Neural vocoder for Qwen3-Omni
//!
//! Converts multi-codebook tokens from Talker into audio waveform.
//!
//! GGUF tensor structure:
//! - code2wav.code_embedding.weight: [1024, 32768] Q8
//! - code2wav.pre_transformer.layers.N.*: 8 layers
//!   - input_layernorm.weight
//!   - self_attn.{q,k,v,o}_proj.weight
//!   - self_attn_layer_scale.scale
//!   - post_attention_layernorm.weight
//!   - mlp.{gate,up,down}_proj.weight
//!   - mlp_layer_scale.scale
//! - code2wav.pre_transformer.norm.weight
//! - code2wav.upsample.N.*: 2 ConvNeXt blocks
//! - code2wav.decoder.N.*: HiFi-GAN style

use super::config::Code2WavConfig;
use super::gguf_loader::Gguf;
use crate::models::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use candle::{DType, Device, Module, Result, Tensor, D};
use std::io::{Read, Seek};
use std::sync::Arc;

/// Snake activation: x + (1/alpha) * sin^2(alpha * x)
/// Used in SnakeBeta: x + (1/beta) * sin^2(alpha * x)
struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    fn new(alpha: Tensor, beta: Tensor) -> Self {
        Self { alpha, beta }
    }
}

impl Module for SnakeBeta {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // x + (1/beta) * sin^2(alpha * x)
        // For Conv1d: xs is [B, C, T], alpha/beta are [C]
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(D::Minus1)?; // [1, C, 1]
        let beta = self.beta.unsqueeze(0)?.unsqueeze(D::Minus1)?;

        let sin_part = (xs.broadcast_mul(&alpha)?.sin())?;
        let sin_sq = (&sin_part * &sin_part)?;
        let scaled = sin_sq.broadcast_div(&beta)?;
        xs + scaled
    }
}

/// Rotary embedding for pre-transformer
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

/// Quantized self-attention for pre-transformer
struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding) -> Result<Tensor> {
        let (b, seq, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = rotary.apply(&q, &k, 0)?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.matmul(&k.t()?)? * scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        let out = out.transpose(1, 2)?.reshape((b, seq, ()))?;
        self.o_proj.forward(&out)
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

/// Pre-transformer layer with layer scales
struct TransformerLayer {
    input_layernorm: RmsNorm,
    self_attn: Attention,
    self_attn_layer_scale: Tensor,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    mlp_layer_scale: Tensor,
}

impl TransformerLayer {
    fn forward(&self, xs: &Tensor, rotary: &RotaryEmbedding) -> Result<Tensor> {
        // Self-attention with residual and layer scale
        let residual = xs;
        let hidden = self.input_layernorm.forward(xs)?;
        let attn_out = self.self_attn.forward(&hidden, rotary)?;
        // Layer scale: element-wise multiply
        let attn_out = attn_out.broadcast_mul(&self.self_attn_layer_scale)?;
        let xs = (residual + attn_out)?;

        // MLP with residual and layer scale
        let residual = &xs;
        let hidden = self.post_attention_layernorm.forward(&xs)?;
        let mlp_out = self.mlp.forward(&hidden)?;
        let mlp_out = mlp_out.broadcast_mul(&self.mlp_layer_scale)?;
        residual + mlp_out
    }
}

/// ConvNeXt block for upsampling
struct ConvNeXtBlock {
    // upsample.N.0: conv_transpose
    conv_transpose: candle_nn::ConvTranspose1d,
    // upsample.N.1: ConvNeXt residual
    dwconv: candle_nn::Conv1d,
    norm: candle_nn::LayerNorm,
    pwconv1: QMatMul,
    pwconv1_bias: Tensor,
    pwconv2: QMatMul,
    pwconv2_bias: Tensor,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // Transpose conv for upsampling
        let xs = self.conv_transpose.forward(xs)?;

        // ConvNeXt block
        let residual = &xs;

        // Depthwise conv
        let hidden = self.dwconv.forward(&xs)?;

        // Transpose [B, C, T] -> [B, T, C] for LayerNorm
        let hidden = hidden.transpose(1, 2)?;
        let hidden = self.norm.forward(&hidden)?;

        // Pointwise convolutions (as linear on [B, T, C])
        let hidden = self.pwconv1.forward(&hidden)?;
        let hidden = hidden.broadcast_add(&self.pwconv1_bias)?;
        let hidden = hidden.gelu_erf()?;
        let hidden = self.pwconv2.forward(&hidden)?;
        let hidden = hidden.broadcast_add(&self.pwconv2_bias)?;

        // Scale by gamma
        let hidden = hidden.broadcast_mul(&self.gamma)?;

        // Transpose back [B, T, C] -> [B, C, T]
        let hidden = hidden.transpose(1, 2)?;

        residual + hidden
    }
}

/// HiFi-GAN residual block with Snake activation
struct ResidualBlock {
    act1: SnakeBeta,
    conv1: candle_nn::Conv1d,
    act2: SnakeBeta,
    conv2: candle_nn::Conv1d,
}

impl ResidualBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let hidden = self.act1.forward(xs)?;
        let hidden = self.conv1.forward(&hidden)?;
        let hidden = self.act2.forward(&hidden)?;
        let hidden = self.conv2.forward(&hidden)?;
        residual + hidden
    }
}

/// Decoder block: Snake activation + upsample conv + residual blocks
struct DecoderBlock {
    snake: SnakeBeta,
    upsample_conv: candle_nn::ConvTranspose1d,
    residual_blocks: Vec<ResidualBlock>,
}

impl DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.snake.forward(xs)?;
        let mut xs = self.upsample_conv.forward(&xs)?;

        for block in &self.residual_blocks {
            xs = block.forward(&xs)?;
        }

        Ok(xs)
    }
}

/// Code2Wav vocoder
pub struct Code2Wav {
    /// Code embedding [vocab_size=32768, embed_dim=1024]
    code_embedding: candle_nn::Embedding,
    /// Pre-transformer layers (8 layers)
    pre_transformer_layers: Vec<TransformerLayer>,
    /// Pre-transformer final norm
    pre_transformer_norm: RmsNorm,
    /// Rotary embedding for attention
    rotary: Arc<RotaryEmbedding>,
    /// Upsample blocks (ConvNeXt)
    upsample_blocks: Vec<ConvNeXtBlock>,
    /// Initial conv before decoder
    initial_conv: candle_nn::Conv1d,
    /// Decoder blocks (HiFi-GAN style)
    decoder_blocks: Vec<DecoderBlock>,
    /// Final snake activation
    final_snake: SnakeBeta,
    /// Final conv to mono
    final_conv: candle_nn::Conv1d,
    /// Config
    #[allow(dead_code)]
    cfg: Code2WavConfig,
    /// Device
    #[allow(dead_code)]
    device: Device,
}

impl Code2Wav {
    /// Load Code2Wav from GGUF file
    pub fn from_gguf<R: Read + Seek>(
        gg: &mut Gguf<R>,
        cfg: &Code2WavConfig,
        device: &Device,
    ) -> Result<Self> {
        let prefix = "code2wav";

        // Code embedding
        let code_embedding = gg.embedding(
            &format!("{prefix}.code_embedding.weight"),
            cfg.embedding_dim,
        )?;

        // Pre-transformer layers
        let mut pre_transformer_layers = Vec::with_capacity(cfg.num_transformer_layers);
        for i in 0..cfg.num_transformer_layers {
            let layer_prefix = format!("{prefix}.pre_transformer.layers.{i}");

            let input_layernorm = gg.rms_norm(
                &format!("{layer_prefix}.input_layernorm.weight"),
                cfg.rms_norm_eps,
            )?;

            let self_attn = Attention {
                q_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.q_proj.weight"))?,
                k_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.k_proj.weight"))?,
                v_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.v_proj.weight"))?,
                o_proj: gg.qmatmul(&format!("{layer_prefix}.self_attn.o_proj.weight"))?,
                num_heads: cfg.num_attention_heads,
                head_dim: cfg.embedding_dim / cfg.num_attention_heads,
            };

            let self_attn_layer_scale =
                gg.dequantize_f32(&format!("{layer_prefix}.self_attn_layer_scale.scale"))?;

            let post_attention_layernorm = gg.rms_norm(
                &format!("{layer_prefix}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )?;

            let mlp = Mlp {
                gate_proj: gg.qmatmul(&format!("{layer_prefix}.mlp.gate_proj.weight"))?,
                up_proj: gg.qmatmul(&format!("{layer_prefix}.mlp.up_proj.weight"))?,
                down_proj: gg.qmatmul(&format!("{layer_prefix}.mlp.down_proj.weight"))?,
            };

            let mlp_layer_scale =
                gg.dequantize_f32(&format!("{layer_prefix}.mlp_layer_scale.scale"))?;

            pre_transformer_layers.push(TransformerLayer {
                input_layernorm,
                self_attn,
                self_attn_layer_scale,
                post_attention_layernorm,
                mlp,
                mlp_layer_scale,
            });
        }

        let pre_transformer_norm = gg.rms_norm(
            &format!("{prefix}.pre_transformer.norm.weight"),
            cfg.rms_norm_eps,
        )?;

        let head_dim = cfg.embedding_dim / cfg.num_attention_heads;
        let rotary = Arc::new(RotaryEmbedding::new(head_dim, 8192, device, DType::F32)?);

        // Upsample blocks (2 ConvNeXt blocks)
        let mut upsample_blocks = Vec::with_capacity(cfg.num_upsample_blocks);
        for i in 0..cfg.num_upsample_blocks {
            let block_prefix = format!("{prefix}.upsample.{i}");

            // Transpose conv for 2x upsampling
            let conv_transpose_cfg = candle_nn::ConvTranspose1dConfig {
                stride: 2,
                padding: 0,
                output_padding: 0,
                dilation: 1,
                groups: 1,
            };
            let conv_transpose = gg.conv_transpose1d(
                &format!("{block_prefix}.0.conv.weight"),
                &format!("{block_prefix}.0.conv.bias"),
                conv_transpose_cfg,
            )?;

            // ConvNeXt residual block
            let dwconv_cfg = candle_nn::Conv1dConfig {
                stride: 1,
                padding: 3, // kernel=7, same padding
                dilation: 1,
                groups: cfg.embedding_dim, // depthwise
                ..Default::default()
            };
            let dwconv = gg.conv1d_depthwise(
                &format!("{block_prefix}.1.dwconv.conv.weight"),
                &format!("{block_prefix}.1.dwconv.conv.bias"),
                dwconv_cfg,
            )?;

            let norm = gg.layer_norm(
                &format!("{block_prefix}.1.norm.weight"),
                &format!("{block_prefix}.1.norm.bias"),
                1e-6,
            )?;

            let pwconv1 = gg.qmatmul(&format!("{block_prefix}.1.pwconv1.weight"))?;
            let pwconv1_bias = gg.dequantize_f32(&format!("{block_prefix}.1.pwconv1.bias"))?;
            let pwconv2 = gg.qmatmul(&format!("{block_prefix}.1.pwconv2.weight"))?;
            let pwconv2_bias = gg.dequantize_f32(&format!("{block_prefix}.1.pwconv2.bias"))?;

            let gamma = gg.dequantize_f32(&format!("{block_prefix}.1.gamma"))?;

            upsample_blocks.push(ConvNeXtBlock {
                conv_transpose,
                dwconv,
                norm,
                pwconv1,
                pwconv1_bias,
                pwconv2,
                pwconv2_bias,
                gamma,
            });
        }

        // Initial conv (decoder.0)
        let initial_conv_cfg = candle_nn::Conv1dConfig {
            stride: 1,
            padding: 3, // kernel=7
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let initial_conv = gg.conv1d(
            &format!("{prefix}.decoder.0.conv.weight"),
            &format!("{prefix}.decoder.0.conv.bias"),
            initial_conv_cfg,
        )?;

        // Decoder blocks (decoder.1 through decoder.4)
        // Each block: snake + transpose conv + 3 residual blocks
        // Channels: 1536 -> 768 -> 384 -> 192 -> 96
        let mut decoder_blocks = Vec::new();
        let upsample_rates = [8, 5, 4, 3]; // Matching HiFi-GAN

        for (i, &rate) in upsample_rates.iter().enumerate() {
            let block_idx = i + 1; // decoder.1, decoder.2, ...
            let block_prefix = format!("{prefix}.decoder.{block_idx}");

            // Snake activation at start of block
            let alpha = gg.dequantize_f32(&format!("{block_prefix}.block.0.alpha"))?;
            let beta = gg.dequantize_f32(&format!("{block_prefix}.block.0.beta"))?;
            let snake = SnakeBeta::new(alpha, beta);

            // Transpose conv for upsampling (kernel_size = rate * 2)
            let upsample_cfg = candle_nn::ConvTranspose1dConfig {
                stride: rate,
                padding: rate / 2,
                output_padding: 0,
                dilation: 1,
                groups: 1,
            };
            let upsample_conv = gg.conv_transpose1d(
                &format!("{block_prefix}.block.1.conv.weight"),
                &format!("{block_prefix}.block.1.conv.bias"),
                upsample_cfg,
            )?;

            // 3 residual blocks with different dilations
            let mut residual_blocks = Vec::new();

            for r in 0..3 {
                let res_idx = r + 2; // block.2, block.3, block.4
                let res_prefix = format!("{block_prefix}.block.{res_idx}");

                let act1_alpha = gg.dequantize_f32(&format!("{res_prefix}.act1.alpha"))?;
                let act1_beta = gg.dequantize_f32(&format!("{res_prefix}.act1.beta"))?;
                let act1 = SnakeBeta::new(act1_alpha, act1_beta);

                let dilation = 3usize.pow(r as u32);
                let conv1_cfg = candle_nn::Conv1dConfig {
                    stride: 1,
                    padding: dilation * 3, // kernel=7
                    dilation,
                    groups: 1,
                    ..Default::default()
                };
                let conv1 = gg.conv1d(
                    &format!("{res_prefix}.conv1.conv.weight"),
                    &format!("{res_prefix}.conv1.conv.bias"),
                    conv1_cfg,
                )?;

                let act2_alpha = gg.dequantize_f32(&format!("{res_prefix}.act2.alpha"))?;
                let act2_beta = gg.dequantize_f32(&format!("{res_prefix}.act2.beta"))?;
                let act2 = SnakeBeta::new(act2_alpha, act2_beta);

                let conv2_cfg = candle_nn::Conv1dConfig {
                    stride: 1,
                    padding: 0, // kernel=1
                    dilation: 1,
                    groups: 1,
                    ..Default::default()
                };
                let conv2 = gg.conv1d(
                    &format!("{res_prefix}.conv2.conv.weight"),
                    &format!("{res_prefix}.conv2.conv.bias"),
                    conv2_cfg,
                )?;

                residual_blocks.push(ResidualBlock {
                    act1,
                    conv1,
                    act2,
                    conv2,
                });
            }

            decoder_blocks.push(DecoderBlock {
                snake,
                upsample_conv,
                residual_blocks,
            });
        }

        // Final snake and conv (decoder.5, decoder.6)
        let final_alpha = gg.dequantize_f32(&format!("{prefix}.decoder.5.alpha"))?;
        let final_beta = gg.dequantize_f32(&format!("{prefix}.decoder.5.beta"))?;
        let final_snake = SnakeBeta::new(final_alpha, final_beta);

        let final_conv_cfg = candle_nn::Conv1dConfig {
            stride: 1,
            padding: 3, // kernel=7
            dilation: 1,
            groups: 1,
            ..Default::default()
        };
        let final_conv = gg.conv1d(
            &format!("{prefix}.decoder.6.conv.weight"),
            &format!("{prefix}.decoder.6.conv.bias"),
            final_conv_cfg,
        )?;

        Ok(Self {
            code_embedding,
            pre_transformer_layers,
            pre_transformer_norm,
            rotary,
            upsample_blocks,
            initial_conv,
            decoder_blocks,
            final_snake,
            final_conv,
            cfg: cfg.clone(),
            device: device.clone(),
        })
    }

    /// Convert codec tokens to audio waveform
    ///
    /// # Arguments
    /// * `codec_tokens` - Token IDs [batch, seq]
    ///
    /// # Returns
    /// * Audio waveform [batch, samples]
    pub fn forward(&self, codec_tokens: &Tensor) -> Result<Tensor> {
        let (_b, _seq) = codec_tokens.dims2()?;

        // Embed tokens: [batch, seq] -> [batch, seq, embed_dim]
        let mut xs = self.code_embedding.forward(codec_tokens)?;

        // Pre-transformer layers
        for layer in &self.pre_transformer_layers {
            xs = layer.forward(&xs, &self.rotary)?;
        }
        xs = self.pre_transformer_norm.forward(&xs)?;

        // Transpose for conv: [B, T, C] -> [B, C, T]
        xs = xs.transpose(1, 2)?;

        // Upsample blocks
        for block in &self.upsample_blocks {
            xs = block.forward(&xs)?;
        }

        // Initial decoder conv
        xs = self.initial_conv.forward(&xs)?;

        // Decoder blocks (HiFi-GAN)
        for block in &self.decoder_blocks {
            xs = block.forward(&xs)?;
        }

        // Final activation and conv
        xs = self.final_snake.forward(&xs)?;
        xs = self.final_conv.forward(&xs)?;

        // Apply tanh for audio normalization
        let xs = xs.tanh()?;

        // Squeeze channel dim: [batch, 1, samples] -> [batch, samples]
        xs.squeeze(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = Code2WavConfig::default();
        assert_eq!(cfg.embedding_dim, 1024);
        assert_eq!(cfg.num_transformer_layers, 8);
        assert_eq!(cfg.num_codebooks, 8);
    }
}
