//! Error types for the wgpu backend

use crate::DType;

/// Errors specific to the wgpu backend
#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("wgpu error: {0}")]
    Message(String),

    #[error("no suitable GPU adapter found")]
    NoAdapter,

    #[error("failed to request device: {0}")]
    DeviceRequest(#[from] wgpu::RequestDeviceError),

    #[error("buffer map failed")]
    BufferMapFailed,

    #[error("unsupported dtype {0:?} for operation {1}")]
    UnsupportedDType(DType, &'static str),

    #[error("shader compilation error: {0}")]
    ShaderCompilation(String),

    #[error("buffer size mismatch: expected {expected}, got {got}")]
    BufferSizeMismatch { expected: usize, got: usize },

    #[error("operation not implemented: {0}")]
    NotImplemented(&'static str),
}

impl From<String> for WgpuError {
    fn from(e: String) -> Self {
        WgpuError::Message(e)
    }
}

impl From<&str> for WgpuError {
    fn from(e: &str) -> Self {
        WgpuError::Message(e.to_string())
    }
}
