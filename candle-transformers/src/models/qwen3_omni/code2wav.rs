//! Code2Wav: Neural vocoder for Qwen3-Omni
//!
//! Converts multi-codebook tokens from Talker into audio waveform.
//! Architecture is similar to HiFi-GAN or Encodec decoder.

use super::config::Code2WavConfig;
use candle::{Module, Result, Tensor};
use candle_nn::VarBuilder;

/// Residual block with dilated convolutions
struct ResidualBlock {
    conv1: candle_nn::Conv1d,
    conv2: candle_nn::Conv1d,
}

impl ResidualBlock {
    fn new(channels: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let padding = dilation;
        let conv1_cfg = candle_nn::Conv1dConfig {
            dilation,
            padding,
            ..Default::default()
        };
        let conv2_cfg = candle_nn::Conv1dConfig {
            padding: 0,
            ..Default::default()
        };

        Ok(Self {
            conv1: candle_nn::conv1d(channels, channels, 3, conv1_cfg, vb.pp("conv1"))?,
            conv2: candle_nn::conv1d(channels, channels, 1, conv2_cfg, vb.pp("conv2"))?,
        })
    }
}

impl Module for ResidualBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let xs = xs.gelu_erf()?;
        let xs = self.conv1.forward(&xs)?;
        let xs = xs.gelu_erf()?;
        let xs = self.conv2.forward(&xs)?;
        residual + xs
    }
}

/// Upsample block with transposed convolution
struct UpsampleBlock {
    conv_trans: candle_nn::ConvTranspose1d,
    residual_blocks: Vec<ResidualBlock>,
}

impl UpsampleBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        upsample_rate: usize,
        num_residual: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = candle_nn::ConvTranspose1dConfig {
            stride: upsample_rate,
            padding: upsample_rate / 2,
            output_padding: 0,
            ..Default::default()
        };

        let kernel_size = upsample_rate * 2;
        let conv_trans = candle_nn::conv_transpose1d(
            in_channels,
            out_channels,
            kernel_size,
            conv_cfg,
            vb.pp("conv_trans"),
        )?;

        let mut residual_blocks = Vec::with_capacity(num_residual);
        let vb_res = vb.pp("residual");
        for i in 0..num_residual {
            let dilation = 3usize.pow(i as u32);
            residual_blocks.push(ResidualBlock::new(out_channels, dilation, vb_res.pp(i))?);
        }

        Ok(Self {
            conv_trans,
            residual_blocks,
        })
    }
}

impl Module for UpsampleBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.gelu_erf()?;
        xs = self.conv_trans.forward(&xs)?;

        for block in &self.residual_blocks {
            xs = block.forward(&xs)?;
        }

        Ok(xs)
    }
}

/// Codebook embedding layer
struct CodebookEmbedding {
    embeddings: Vec<candle_nn::Embedding>,
    proj: candle_nn::Conv1d,
}

impl CodebookEmbedding {
    fn new(cfg: &Code2WavConfig, hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let mut embeddings = Vec::with_capacity(cfg.num_codebooks);
        let vb_emb = vb.pp("codebook_embeddings");

        // Each codebook has its own embedding
        for i in 0..cfg.num_codebooks {
            embeddings.push(candle_nn::embedding(
                cfg.codebook_size,
                hidden_size / cfg.num_codebooks,
                vb_emb.pp(i),
            )?);
        }

        // Project concatenated embeddings
        let proj_cfg = candle_nn::Conv1dConfig::default();
        let proj = candle_nn::conv1d(hidden_size, hidden_size, 1, proj_cfg, vb.pp("proj"))?;

        Ok(Self { embeddings, proj })
    }

    /// Embed codec tokens from all codebooks
    /// Input: [batch, seq, num_codebooks]
    /// Output: [batch, hidden, seq]
    fn forward(&self, tokens: &Tensor) -> Result<Tensor> {
        let (_batch, _seq, _num_cb) = tokens.dims3()?;
        let mut cb_embeddings = Vec::with_capacity(self.embeddings.len());

        for (i, emb) in self.embeddings.iter().enumerate() {
            let cb_tokens = tokens.narrow(2, i, 1)?.squeeze(2)?;
            let embedded = emb.forward(&cb_tokens)?;
            cb_embeddings.push(embedded);
        }

        // Concatenate along hidden dim: [batch, seq, hidden]
        let combined = Tensor::cat(&cb_embeddings, 2)?;

        // Transpose for conv: [batch, hidden, seq]
        let combined = combined.transpose(1, 2)?;

        self.proj.forward(&combined)
    }
}

/// Code2Wav: Vocoder for audio synthesis
pub struct Code2Wav {
    codebook_embed: CodebookEmbedding,
    upsample_blocks: Vec<UpsampleBlock>,
    output_conv: candle_nn::Conv1d,
}

impl Code2Wav {
    pub fn new(cfg: &Code2WavConfig, vb: VarBuilder) -> Result<Self> {
        let codebook_embed = CodebookEmbedding::new(cfg, cfg.hidden_size, vb.pp("codebook_embed"))?;

        let mut upsample_blocks = Vec::with_capacity(cfg.num_upsample_layers);
        let vb_up = vb.pp("upsample");

        let mut current_channels = cfg.hidden_size;
        for (i, &upsample_rate) in cfg.upsample_rates.iter().enumerate() {
            let out_channels = current_channels / 2;
            upsample_blocks.push(UpsampleBlock::new(
                current_channels,
                out_channels,
                upsample_rate,
                cfg.num_residual_blocks,
                vb_up.pp(i),
            )?);
            current_channels = out_channels;
        }

        // Final output conv: project to 1 channel (mono audio)
        let output_cfg = candle_nn::Conv1dConfig {
            padding: 3,
            ..Default::default()
        };
        let output_conv = candle_nn::conv1d(current_channels, 1, 7, output_cfg, vb.pp("output_conv"))?;

        Ok(Self {
            codebook_embed,
            upsample_blocks,
            output_conv,
        })
    }

    /// Convert codec tokens to audio waveform
    ///
    /// # Arguments
    /// * `codec_tokens` - Multi-codebook tokens [batch, seq, num_codebooks]
    ///
    /// # Returns
    /// * Audio waveform [batch, samples]
    pub fn forward(&self, codec_tokens: &Tensor) -> Result<Tensor> {
        // Embed codec tokens: [batch, hidden, seq]
        let mut xs = self.codebook_embed.forward(codec_tokens)?;

        // Upsample through blocks
        for block in &self.upsample_blocks {
            xs = block.forward(&xs)?;
        }

        // Final activation and output projection
        let xs = xs.gelu_erf()?;
        let xs = self.output_conv.forward(&xs)?;

        // Apply tanh for audio normalization
        let xs = xs.tanh()?;

        // Squeeze channel dim: [batch, 1, samples] -> [batch, samples]
        xs.squeeze(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    #[test]
    fn test_config_defaults() {
        let cfg = Code2WavConfig::default();
        assert_eq!(cfg.num_codebooks, 4);
        assert_eq!(cfg.upsample_rates, vec![8, 5, 4, 2]);

        // Total upsample factor should be 320 (50Hz codec -> 16kHz audio)
        let total: usize = cfg.upsample_rates.iter().product();
        assert_eq!(total, 320);
    }
}
