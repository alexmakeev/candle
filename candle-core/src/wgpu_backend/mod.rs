//! wgpu/Vulkan backend for Candle ML framework
//!
//! This module provides GPU compute support through wgpu,
//! which abstracts over Vulkan, Metal, DX12, and WebGPU backends.

mod device;
mod error;
pub mod ops;
pub mod quantized;
mod storage;

pub use device::{CachedPipeline, DeviceId, ShaderType, WgpuDevice};
pub use error::WgpuError;
pub use storage::WgpuStorage;

/// Check if wgpu is available on this system
pub fn is_available() -> bool {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    !instance.enumerate_adapters(wgpu::Backends::all()).is_empty()
}

/// Get information about available wgpu adapters
pub fn list_adapters() -> Vec<wgpu::AdapterInfo> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });

    instance
        .enumerate_adapters(wgpu::Backends::all())
        .into_iter()
        .map(|a| a.get_info())
        .collect()
}
