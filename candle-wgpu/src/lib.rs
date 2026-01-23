//! wgpu/Vulkan backend for Candle ML framework
//!
//! This crate provides GPU compute support for Candle through wgpu,
//! which abstracts over Vulkan, Metal, DX12, and WebGPU backends.
//!
//! ## Features
//!
//! - Cross-platform GPU compute (Vulkan on Linux/Windows, Metal on macOS)
//! - Automatic fallback to CPU when no GPU is available
//! - Support for common ML operations (matmul, softmax, etc.)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use candle_wgpu::WgpuDevice;
//! use candle_core::backend::BackendDevice;
//!
//! // Create a wgpu device (selects best available GPU)
//! let device = WgpuDevice::new(0)?;
//!
//! // Check device info
//! let info = device.adapter_info();
//! println!("Using GPU: {} ({:?})", info.name, info.backend);
//! ```
//!
//! ## Current Status
//!
//! This is an early implementation. Currently, most operations fall back to CPU.
//! The following operations have native GPU implementations:
//!
//! - TODO: matmul (in progress)
//! - TODO: softmax (in progress)
//! - TODO: layer_norm (planned)
//!

mod device;
mod error;
mod storage;

pub mod ops;

pub use device::{DeviceId, WgpuDevice};
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::backend::{BackendDevice, BackendStorage};
    use candle_core::{DType, Shape};

    #[test]
    fn test_device_creation() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let info = device.adapter_info();
        println!("Device: {} ({:?})", info.name, info.backend);
    }

    #[test]
    fn test_zeros() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let shape = Shape::from((2, 3));
        let storage = device.zeros_impl(&shape, DType::F32).expect("Failed to create zeros");

        let cpu_storage = storage.to_cpu_storage().expect("Failed to transfer to CPU");
        match cpu_storage {
            candle_core::CpuStorage::F32(data) => {
                assert_eq!(data.len(), 6);
                assert!(data.iter().all(|&x| x == 0.0));
            }
            _ => panic!("Unexpected dtype"),
        }
    }

    #[test]
    fn test_from_slice() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let storage = device.storage_from_slice(&data).expect("Failed to create storage");

        let cpu_storage = storage.to_cpu_storage().expect("Failed to transfer to CPU");
        match cpu_storage {
            candle_core::CpuStorage::F32(result) => {
                assert_eq!(result, data);
            }
            _ => panic!("Unexpected dtype"),
        }
    }

    #[test]
    fn test_list_adapters() {
        let adapters = list_adapters();
        for (i, info) in adapters.iter().enumerate() {
            println!("Adapter {}: {} ({:?})", i, info.name, info.backend);
        }
    }
}
