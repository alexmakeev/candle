//! GGUF loader utilities for Qwen3-Omni
//!
//! Provides helpers for loading quantized tensors from GGUF files.

use super::config::Config;
use crate::models::with_tracing::QMatMul;
use crate::quantized_nn::RmsNorm;
use candle::quantized::{gguf_file, QTensor};
use candle::{DType, Device, Result, Tensor};
use candle_nn::Embedding;
use std::io::{Read, Seek};

/// GGUF file loader with tensor access helpers
pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    /// Create loader from GGUF content
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get metadata
    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    /// Get raw QTensor by name
    pub fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }

    /// Get QMatMul (quantized linear layer)
    pub fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.tensor(name)?;
        QMatMul::from_weights(ws.into())
    }

    /// Get RmsNorm from quantized weight
    pub fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.tensor(name)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    /// Get LayerNorm (dequantized)
    pub fn layer_norm(&mut self, weight_name: &str, bias_name: &str, eps: f64) -> Result<candle_nn::LayerNorm> {
        let weight = self.tensor(weight_name)?.dequantize(&self.device)?;
        let bias = self.tensor(bias_name)?.dequantize(&self.device)?;
        Ok(candle_nn::LayerNorm::new(weight, bias, eps))
    }

    /// Get LayerNorm without bias (dequantized)
    pub fn layer_norm_no_bias(&mut self, name: &str, eps: f64) -> Result<candle_nn::LayerNorm> {
        let weight = self.tensor(name)?.dequantize(&self.device)?;
        Ok(candle_nn::LayerNorm::new_no_bias(weight, eps))
    }

    /// Get Embedding (dequantized for lookup table)
    pub fn embedding(&mut self, name: &str, hidden_size: usize) -> Result<Embedding> {
        let ws = self.tensor(name)?.dequantize(&self.device)?;
        Ok(Embedding::new(ws, hidden_size))
    }

    /// Get dequantized tensor as F32
    pub fn dequantize_f32(&mut self, name: &str) -> Result<Tensor> {
        let qt = self.tensor(name)?;
        qt.dequantize(&self.device)?.to_dtype(DType::F32)
    }

    /// Get dequantized tensor preserving dtype
    pub fn dequantize(&mut self, name: &str) -> Result<Tensor> {
        self.tensor(name)?.dequantize(&self.device)
    }

    /// Check if tensor exists
    pub fn has_tensor(&self, name: &str) -> bool {
        self.ct.tensor_infos.contains_key(name)
    }

    /// Get optional tensor
    pub fn tensor_opt(&mut self, name: &str) -> Option<QTensor> {
        if self.has_tensor(name) {
            self.tensor(name).ok()
        } else {
            None
        }
    }

    /// Get optional QMatMul
    pub fn qmatmul_opt(&mut self, name: &str) -> Option<QMatMul> {
        if self.has_tensor(name) {
            self.qmatmul(name).ok()
        } else {
            None
        }
    }

    /// Create config from GGUF metadata (if available)
    /// For now returns default config since qwen3-omni GGUF has no metadata
    pub fn config(&self) -> Result<Config> {
        // GGUF v2 has 0 metadata entries, use defaults
        Ok(Config::default())
    }

    /// Get Conv1d weights
    ///
    /// GGUF/Candle dequantize returns [out_ch, in_ch, kernel] format
    /// which is exactly what Candle Conv1d expects.
    pub fn conv1d(
        &mut self,
        weight_name: &str,
        bias_name: &str,
        config: candle_nn::Conv1dConfig,
    ) -> Result<candle_nn::Conv1d> {
        let weight = self.dequantize(weight_name)?;
        let bias = self.dequantize(bias_name)?;

        // Weight is already in [out_ch, in_ch, kernel] format
        Ok(candle_nn::Conv1d::new(weight, Some(bias), config))
    }

    /// Get Conv1d with depthwise groups
    ///
    /// GGUF/Candle dequantize returns [channels, 1, kernel] format
    /// which is exactly what Candle Conv1d expects for depthwise.
    pub fn conv1d_depthwise(
        &mut self,
        weight_name: &str,
        bias_name: &str,
        config: candle_nn::Conv1dConfig,
    ) -> Result<candle_nn::Conv1d> {
        let weight = self.dequantize(weight_name)?;
        let bias = self.dequantize(bias_name)?;

        // Weight is already in [channels, 1, kernel] format
        Ok(candle_nn::Conv1d::new(weight, Some(bias), config))
    }

    /// Get ConvTranspose1d weights
    ///
    /// GGUF/Candle dequantize returns weights in [in_ch, out_ch, kernel] format
    /// which is exactly what Candle ConvTranspose1d expects.
    pub fn conv_transpose1d(
        &mut self,
        weight_name: &str,
        bias_name: &str,
        config: candle_nn::ConvTranspose1dConfig,
    ) -> Result<candle_nn::ConvTranspose1d> {
        let weight = self.dequantize(weight_name)?;
        let bias = self.dequantize(bias_name)?;

        // Weight is already in [in_ch, out_ch, kernel] format
        Ok(candle_nn::ConvTranspose1d::new(weight, Some(bias), config))
    }
}
