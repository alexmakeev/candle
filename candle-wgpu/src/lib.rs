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
pub mod quantized;
pub mod softmax;
pub mod layer_norm;
pub mod rope;

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

    #[test]
    fn test_matmul_simple() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let info = device.adapter_info();
        println!("Testing matmul on: {} ({:?})", info.name, info.backend);

        // Create test matrices A[2,3] @ B[3,4] = C[2,4]
        let a_data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let b_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];

        // Expected result (manually calculated):
        // C[0,0] = 1*1 + 2*5 + 3*9 = 1 + 10 + 27 = 38
        // C[0,1] = 1*2 + 2*6 + 3*10 = 2 + 12 + 30 = 44
        // C[0,2] = 1*3 + 2*7 + 3*11 = 3 + 14 + 33 = 50
        // C[0,3] = 1*4 + 2*8 + 3*12 = 4 + 16 + 36 = 56
        // C[1,0] = 4*1 + 5*5 + 6*9 = 4 + 25 + 54 = 83
        // C[1,1] = 4*2 + 5*6 + 6*10 = 8 + 30 + 60 = 98
        // C[1,2] = 4*3 + 5*7 + 6*11 = 12 + 35 + 66 = 113
        // C[1,3] = 4*4 + 5*8 + 6*12 = 16 + 40 + 72 = 128
        let expected: Vec<f32> = vec![
            38.0, 44.0, 50.0, 56.0,
            83.0, 98.0, 113.0, 128.0,
        ];

        // Create storage
        let a_storage = device.storage_from_slice(&a_data).expect("Failed to create A storage");
        let b_storage = device.storage_from_slice(&b_data).expect("Failed to create B storage");

        // Create layouts for the matrices
        let a_shape = candle_core::Shape::from((2, 3));
        let b_shape = candle_core::Shape::from((3, 4));
        let a_layout = candle_core::Layout::contiguous(&a_shape);
        let b_layout = candle_core::Layout::contiguous(&b_shape);

        // Perform matmul: bmnk = (batch=1, m=2, n=4, k=3)
        let c_storage = a_storage
            .matmul(&b_storage, (1, 2, 4, 3), &a_layout, &b_layout)
            .expect("matmul failed");

        // Read back result
        let c_cpu = c_storage.to_cpu_storage().expect("Failed to transfer result to CPU");
        match c_cpu {
            candle_core::CpuStorage::F32(result) => {
                println!("Result: {:?}", result);
                println!("Expected: {:?}", expected);
                assert_eq!(result.len(), expected.len(), "Result length mismatch");
                for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (r - e).abs() < 1e-5,
                        "Mismatch at index {}: got {}, expected {}",
                        i, r, e
                    );
                }
                println!("matmul F32 test passed!");
            }
            _ => panic!("Unexpected dtype in result"),
        }
    }

    #[test]
    fn test_bf16_matmul_simple() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let info = device.adapter_info();
        println!("Testing BF16 matmul on: {} ({:?})", info.name, info.backend);

        // Create test matrices A[2,3] @ B[3,4] = C[2,4]
        // Using simple values that convert cleanly to BF16
        let a_f32: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let b_f32: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];

        // Convert to BF16 (as u16 bits)
        let a_bf16: Vec<half::bf16> = a_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let b_bf16: Vec<half::bf16> = b_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

        // Expected result (same as F32 test):
        // C[0,0] = 1*1 + 2*5 + 3*9 = 38
        // C[0,1] = 1*2 + 2*6 + 3*10 = 44
        // C[0,2] = 1*3 + 2*7 + 3*11 = 50
        // C[0,3] = 1*4 + 2*8 + 3*12 = 56
        // C[1,0] = 4*1 + 5*5 + 6*9 = 83
        // C[1,1] = 4*2 + 5*6 + 6*10 = 98
        // C[1,2] = 4*3 + 5*7 + 6*11 = 113
        // C[1,3] = 4*4 + 5*8 + 6*12 = 128
        let expected: Vec<f32> = vec![
            38.0, 44.0, 50.0, 56.0,
            83.0, 98.0, 113.0, 128.0,
        ];

        // Create BF16 storage
        // BF16 is stored as raw u16 bytes
        let a_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                a_bf16.as_ptr() as *const u8,
                a_bf16.len() * 2,
            )
        };
        let b_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                b_bf16.as_ptr() as *const u8,
                b_bf16.len() * 2,
            )
        };

        let a_buffer = device.create_buffer_init(
            a_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "bf16_a",
        );
        let b_buffer = device.create_buffer_init(
            b_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "bf16_b",
        );

        let a_storage = WgpuStorage::new(
            std::sync::Arc::new(a_buffer),
            device.clone(),
            a_bf16.len(),
            DType::BF16,
        );
        let b_storage = WgpuStorage::new(
            std::sync::Arc::new(b_buffer),
            device.clone(),
            b_bf16.len(),
            DType::BF16,
        );

        // Create layouts
        let a_shape = candle_core::Shape::from((2, 3));
        let b_shape = candle_core::Shape::from((3, 4));
        let a_layout = candle_core::Layout::contiguous(&a_shape);
        let b_layout = candle_core::Layout::contiguous(&b_shape);

        // Perform matmul: bmnk = (batch=1, m=2, n=4, k=3)
        let c_storage = a_storage
            .matmul(&b_storage, (1, 2, 4, 3), &a_layout, &b_layout)
            .expect("BF16 matmul failed");

        // Read back result (BF16 -> f32)
        let c_cpu = c_storage.to_cpu_storage().expect("Failed to transfer result to CPU");
        match c_cpu {
            candle_core::CpuStorage::BF16(result_bf16) => {
                let result: Vec<f32> = result_bf16.iter().map(|x| x.to_f32()).collect();
                println!("Result: {:?}", result);
                println!("Expected: {:?}", expected);
                assert_eq!(result.len(), expected.len(), "Result length mismatch");
                for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
                    // BF16 has lower precision, allow 1% error
                    let rel_error = if e.abs() > 1e-6 { (r - e).abs() / e.abs() } else { (r - e).abs() };
                    assert!(
                        rel_error < 0.01,
                        "Mismatch at index {}: got {}, expected {}, rel_error={}%",
                        i, r, e, rel_error * 100.0
                    );
                }
                println!("BF16 matmul test passed!");
            }
            _ => panic!("Unexpected dtype in result: expected BF16"),
        }
    }

    #[test]
    fn test_bf16_matmul_larger() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing larger BF16 matmul on: {}", device.adapter_info().name);

        // 32x64 @ 64x32 = 32x32
        let m = 32;
        let k = 64;
        let n = 32;

        // Create test data with larger values to avoid near-zero relative error issues
        // Values range roughly from -8 to +8
        let a_f32: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 - 8.0).collect();
        let b_f32: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect();

        // Convert to BF16 first, then compute expected from BF16 values
        // This gives us the "fair" expected result accounting for BF16 input quantization
        let a_bf16: Vec<half::bf16> = a_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let b_bf16: Vec<half::bf16> = b_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

        // Compute expected on CPU using BF16 inputs
        let mut expected = vec![0.0f32; m * n];
        for row in 0..m {
            for col in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    let a_val = a_bf16[row * k + kk].to_f32();
                    let b_val = b_bf16[kk * n + col].to_f32();
                    sum += a_val * b_val;
                }
                expected[row * n + col] = sum;
            }
        }

        let a_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(a_bf16.as_ptr() as *const u8, a_bf16.len() * 2)
        };
        let b_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(b_bf16.as_ptr() as *const u8, b_bf16.len() * 2)
        };

        let a_buffer = device.create_buffer_init(
            a_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "bf16_a_large",
        );
        let b_buffer = device.create_buffer_init(
            b_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "bf16_b_large",
        );

        let a_storage = WgpuStorage::new(
            std::sync::Arc::new(a_buffer),
            device.clone(),
            a_bf16.len(),
            DType::BF16,
        );
        let b_storage = WgpuStorage::new(
            std::sync::Arc::new(b_buffer),
            device.clone(),
            b_bf16.len(),
            DType::BF16,
        );

        let a_shape = candle_core::Shape::from((m, k));
        let b_shape = candle_core::Shape::from((k, n));
        let a_layout = candle_core::Layout::contiguous(&a_shape);
        let b_layout = candle_core::Layout::contiguous(&b_shape);

        let c_storage = a_storage
            .matmul(&b_storage, (1, m, n, k), &a_layout, &b_layout)
            .expect("BF16 matmul failed");

        let c_cpu = c_storage.to_cpu_storage().expect("Failed to transfer");
        match c_cpu {
            candle_core::CpuStorage::BF16(result_bf16) => {
                let result: Vec<f32> = result_bf16.iter().map(|x| x.to_f32()).collect();

                // Check max error
                let mut max_abs_error = 0.0f32;
                let mut max_rel_error = 0.0f32;
                let mut worst_idx = 0;
                for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
                    let abs_error = (r - e).abs();
                    // For relative error, only consider values with reasonable magnitude
                    let rel_error = if e.abs() > 1.0 { abs_error / e.abs() } else { abs_error / 1.0 };
                    if abs_error > max_abs_error {
                        max_abs_error = abs_error;
                    }
                    if rel_error > max_rel_error {
                        max_rel_error = rel_error;
                        worst_idx = i;
                    }
                }
                println!("Max absolute error: {:.6}", max_abs_error);
                println!("Max relative error: {:.4}% at index {} (got={}, expected={})",
                        max_rel_error * 100.0, worst_idx, result[worst_idx], expected[worst_idx]);

                // BF16 has 7 mantissa bits, so ~0.8% precision
                // With 64 accumulations, error can compound
                // Allow 2% relative error for large values
                assert!(max_rel_error < 0.02 || max_abs_error < 1.0,
                        "Error too large: rel={:.2}%, abs={:.4}",
                        max_rel_error * 100.0, max_abs_error);
                println!("Larger BF16 matmul test passed! ({}x{} @ {}x{})", m, k, k, n);
            }
            _ => panic!("Unexpected dtype"),
        }
    }
}
