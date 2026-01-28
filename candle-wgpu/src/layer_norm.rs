//! Layer normalization for wgpu backend
//!
//! Implements layer normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta

use crate::device::{ShaderType, WgpuDevice};
use crate::storage::WgpuStorage;
use candle_core::backend::BackendStorage;
use candle_core::DType;
use std::sync::Arc;

/// Parameters for layer normalization
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LayerNormParams {
    /// Batch size (number of vectors to normalize)
    pub batch_size: u32,
    /// Hidden size (size of each vector)
    pub hidden_size: u32,
    /// Epsilon for numerical stability
    pub eps: f32,
    /// Padding to 16-byte alignment
    pub _padding: u32,
}

/// GPU-accelerated layer normalization
///
/// Computes: y = (x - mean) / sqrt(var + eps) * gamma + beta
///
/// # Arguments
/// * `device` - The wgpu device
/// * `input` - Input tensor [batch_size, hidden_size]
/// * `gamma` - Scale parameter [hidden_size]
/// * `beta` - Bias parameter [hidden_size]
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// Normalized output [batch_size, hidden_size]
pub fn layer_norm_gpu(
    device: &WgpuDevice,
    input: &WgpuStorage,
    gamma: &WgpuStorage,
    beta: &WgpuStorage,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) -> candle_core::Result<WgpuStorage> {
    assert_eq!(input.dtype(), DType::F32, "layer_norm_gpu only supports F32");
    assert_eq!(gamma.dtype(), DType::F32);
    assert_eq!(beta.dtype(), DType::F32);
    assert_eq!(input.count(), batch_size * hidden_size);
    assert_eq!(gamma.count(), hidden_size);
    assert_eq!(beta.count(), hidden_size);

    let output_size = batch_size * hidden_size;
    let output_bytes = output_size * std::mem::size_of::<f32>();

    // Create output buffer
    let output_buffer = device.create_buffer(
        output_bytes as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        "layer_norm_output",
    );

    // Create params buffer
    let params = LayerNormParams {
        batch_size: batch_size as u32,
        hidden_size: hidden_size as u32,
        eps,
        _padding: 0,
    };
    let params_bytes = bytemuck::bytes_of(&params);
    let params_buffer = device.create_buffer_init(
        params_bytes,
        wgpu::BufferUsages::UNIFORM,
        "layer_norm_params",
    );

    // Execute layer norm shader
    device.with_pipeline(ShaderType::LayerNormF32, |cached| {
        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bind_group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: beta.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer_norm_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layer_norm_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Each thread processes one batch element
            let workgroups = (batch_size as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.queue().submit(std::iter::once(encoder.finish()));
    });

    Ok(WgpuStorage::new(
        Arc::new(output_buffer),
        device.clone(),
        output_size,
        DType::F32,
    ))
}

/// GPU-accelerated BF16 layer normalization
///
/// Input: BF16 stored as u16
/// Computation: Convert to F32, compute layer norm, return F32 (to be converted to BF16 on CPU)
///
/// # Arguments
/// * `device` - The wgpu device
/// * `input` - Input tensor [batch_size, hidden_size] (BF16 as u16)
/// * `gamma` - Scale parameter [hidden_size] (BF16 as u16)
/// * `beta` - Bias parameter [hidden_size] (BF16 as u16)
/// * `eps` - Epsilon for numerical stability
///
/// # Returns
/// Normalized output [batch_size, hidden_size] in F32
pub fn layer_norm_bf16_gpu(
    device: &WgpuDevice,
    input: &WgpuStorage,
    gamma: &WgpuStorage,
    beta: &WgpuStorage,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) -> candle_core::Result<WgpuStorage> {
    use candle_core::DType;

    assert_eq!(input.dtype(), DType::BF16, "layer_norm_bf16_gpu requires BF16 input");
    assert_eq!(gamma.dtype(), DType::BF16);
    assert_eq!(beta.dtype(), DType::BF16);
    assert_eq!(input.count(), batch_size * hidden_size);
    assert_eq!(gamma.count(), hidden_size);
    assert_eq!(beta.count(), hidden_size);

    let output_size = batch_size * hidden_size;
    let output_bytes = output_size * std::mem::size_of::<f32>();

    // Create F32 output buffer
    let output_buffer = device.create_buffer(
        output_bytes as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        "layer_norm_bf16_output",
    );

    // Create params buffer
    let params = LayerNormParams {
        batch_size: batch_size as u32,
        hidden_size: hidden_size as u32,
        eps,
        _padding: 0,
    };
    let params_bytes = bytemuck::bytes_of(&params);
    let params_buffer = device.create_buffer_init(
        params_bytes,
        wgpu::BufferUsages::UNIFORM,
        "layer_norm_bf16_params",
    );

    // Execute BF16 layer norm shader
    device.with_pipeline(ShaderType::LayerNormBF16, |cached| {
        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("layer_norm_bf16_bind_group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gamma.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: beta.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("layer_norm_bf16_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("layer_norm_bf16_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Each thread processes one batch element
            let workgroups = (batch_size as u32 + 255) / 256;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.queue().submit(std::iter::once(encoder.finish()));
    });

    // Return F32 output (caller can convert to BF16 if needed)
    Ok(WgpuStorage::new(
        Arc::new(output_buffer),
        device.clone(),
        output_size,
        DType::F32,
    ))
}

/// CPU reference implementation for testing
#[cfg(test)]
fn cpu_layer_norm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0f32; input.len()];

    for b in 0..batch_size {
        let offset = b * hidden_size;
        let row = &input[offset..offset + hidden_size];

        // Compute mean
        let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;

        // Compute variance
        let variance: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;

        // Normalize
        let std_inv = 1.0 / (variance + eps).sqrt();
        for i in 0..hidden_size {
            let normalized = (row[i] - mean) * std_inv;
            output[offset + i] = normalized * gamma[i] + beta[i];
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{is_available, WgpuDevice};
    use candle_core::backend::BackendDevice;

    #[test]
    fn test_layer_norm_simple() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let info = device.adapter_info();
        println!("Testing layer norm on: {} ({:?})", info.name, info.backend);

        let batch_size = 2;
        let hidden_size = 4;
        let eps = 1e-5;

        // Input: 2 batches, 4 features each
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
        ];

        // Gamma and beta (scale and shift)
        let gamma: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0]; // No scaling
        let beta: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];  // No shift

        // Create storage
        let input_storage = device.storage_from_slice(&input_data).expect("Failed to create input");
        let gamma_storage = device.storage_from_slice(&gamma).expect("Failed to create gamma");
        let beta_storage = device.storage_from_slice(&beta).expect("Failed to create beta");

        // Run GPU layer norm
        let output_storage = layer_norm_gpu(
            &device, &input_storage, &gamma_storage, &beta_storage,
            batch_size, hidden_size, eps
        ).expect("layer_norm failed");

        // Read back result
        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        // Compute expected
        let expected = cpu_layer_norm(&input_data, &gamma, &beta, batch_size, hidden_size, eps);

        println!("Input: {:?}", input_data);
        println!("GPU output: {:?}", output);
        println!("CPU expected: {:?}", expected);

        // Verify results
        for i in 0..output.len() {
            let error = (output[i] - expected[i]).abs();
            assert!(
                error < 1e-5,
                "Mismatch at index {}: got {}, expected {}, error {}",
                i, output[i], expected[i], error
            );
        }

        // Verify layer norm properties: each row should have mean ≈ 0, std ≈ 1
        for b in 0..batch_size {
            let offset = b * hidden_size;
            let row = &output[offset..offset + hidden_size];

            let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
            let variance: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;

            assert!(
                mean.abs() < 1e-5,
                "Row {} mean should be ~0, got {}",
                b, mean
            );
            assert!(
                (variance - 1.0).abs() < 1e-4,
                "Row {} variance should be ~1, got {}",
                b, variance
            );
        }

        println!("✅ Layer norm test passed!");
    }

    #[test]
    fn test_layer_norm_with_scale_shift() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing layer norm with scale/shift on: {}", device.adapter_info().name);

        let batch_size = 2;
        let hidden_size = 4;
        let eps = 1e-5;

        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            -1.0, 0.0, 1.0, 2.0,
        ];

        // Non-trivial gamma and beta
        let gamma: Vec<f32> = vec![2.0, 0.5, 1.0, 3.0];
        let beta: Vec<f32> = vec![1.0, -1.0, 0.0, 2.0];

        let input_storage = device.storage_from_slice(&input_data).expect("Failed to create input");
        let gamma_storage = device.storage_from_slice(&gamma).expect("Failed to create gamma");
        let beta_storage = device.storage_from_slice(&beta).expect("Failed to create beta");

        let output_storage = layer_norm_gpu(
            &device, &input_storage, &gamma_storage, &beta_storage,
            batch_size, hidden_size, eps
        ).expect("layer_norm failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        let expected = cpu_layer_norm(&input_data, &gamma, &beta, batch_size, hidden_size, eps);

        println!("GPU output: {:?}", output);
        println!("CPU expected: {:?}", expected);

        for i in 0..output.len() {
            let error = (output[i] - expected[i]).abs();
            assert!(
                error < 1e-4,
                "Mismatch at index {}: got {}, expected {}, error {}",
                i, output[i], expected[i], error
            );
        }

        println!("✅ Layer norm with scale/shift test passed!");
    }

    #[test]
    fn test_layer_norm_large() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing large layer norm on: {}", device.adapter_info().name);

        // Typical transformer dimensions
        let batch_size = 32;
        let hidden_size = 768;
        let eps = 1e-5;

        // Random-ish data
        let input_data: Vec<f32> = (0..batch_size * hidden_size)
            .map(|i| ((i as f32 * 0.1234).sin() + (i as f32 * 0.5678).cos()) * 2.0)
            .collect();

        let gamma: Vec<f32> = (0..hidden_size).map(|i| 0.5 + (i as f32 / hidden_size as f32)).collect();
        let beta: Vec<f32> = (0..hidden_size).map(|i| (i as f32 / hidden_size as f32) - 0.5).collect();

        let input_storage = device.storage_from_slice(&input_data).expect("Failed to create input");
        let gamma_storage = device.storage_from_slice(&gamma).expect("Failed to create gamma");
        let beta_storage = device.storage_from_slice(&beta).expect("Failed to create beta");

        let output_storage = layer_norm_gpu(
            &device, &input_storage, &gamma_storage, &beta_storage,
            batch_size, hidden_size, eps
        ).expect("layer_norm failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        let expected = cpu_layer_norm(&input_data, &gamma, &beta, batch_size, hidden_size, eps);

        // Check max error
        let mut max_error = 0.0f32;
        for i in 0..output.len() {
            max_error = max_error.max((output[i] - expected[i]).abs());
        }
        println!("Max error: {:.6e}", max_error);
        assert!(max_error < 1e-3, "Max error too large: {}", max_error);

        println!("✅ Large layer norm test passed! ({} × {})", batch_size, hidden_size);
    }

    #[test]
    fn test_bf16_layer_norm() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let info = device.adapter_info();
        println!("Testing BF16 layer norm on: {} ({:?})", info.name, info.backend);

        let batch_size = 2;
        let hidden_size = 4;
        let eps = 1e-5;

        // Input data in F32
        let input_f32: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
        ];

        let gamma_f32: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
        let beta_f32: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0];

        // Convert to BF16
        let input_bf16: Vec<half::bf16> = input_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let gamma_bf16: Vec<half::bf16> = gamma_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let beta_bf16: Vec<half::bf16> = beta_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

        // Create storage as BF16
        let input_storage = device.storage_from_slice(&input_bf16).expect("Failed to create input");
        let gamma_storage = device.storage_from_slice(&gamma_bf16).expect("Failed to create gamma");
        let beta_storage = device.storage_from_slice(&beta_bf16).expect("Failed to create beta");

        // Run GPU BF16 layer norm (returns F32)
        let output_storage = layer_norm_bf16_gpu(
            &device, &input_storage, &gamma_storage, &beta_storage,
            batch_size, hidden_size, eps
        ).expect("bf16 layer_norm failed");

        // Read back result (F32)
        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        // Compute expected (using F32 reference)
        let expected = cpu_layer_norm(&input_f32, &gamma_f32, &beta_f32, batch_size, hidden_size, eps);

        println!("Input F32: {:?}", input_f32);
        println!("GPU BF16 output: {:?}", output);
        println!("CPU F32 expected: {:?}", expected);

        // Verify results (BF16 has less precision)
        for i in 0..output.len() {
            let error = (output[i] - expected[i]).abs();
            assert!(
                error < 1e-2,
                "Mismatch at index {}: got {}, expected {}, error {}",
                i, output[i], expected[i], error
            );
        }

        // Verify layer norm properties
        for b in 0..batch_size {
            let offset = b * hidden_size;
            let row = &output[offset..offset + hidden_size];

            let mean: f32 = row.iter().sum::<f32>() / hidden_size as f32;
            let variance: f32 = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_size as f32;

            assert!(
                mean.abs() < 1e-2,
                "Row {} mean should be ~0, got {}",
                b, mean
            );
            assert!(
                (variance - 1.0).abs() < 1e-2,
                "Row {} variance should be ~1, got {}",
                b, variance
            );
        }

        println!("✅ BF16 layer norm test passed!");
    }

    #[test]
    fn test_bf16_layer_norm_with_scale_shift() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing BF16 layer norm with scale/shift on: {}", device.adapter_info().name);

        let batch_size = 2;
        let hidden_size = 4;
        let eps = 1e-5;

        let input_f32: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            -1.0, 0.0, 1.0, 2.0,
        ];

        let gamma_f32: Vec<f32> = vec![2.0, 0.5, 1.0, 3.0];
        let beta_f32: Vec<f32> = vec![1.0, -1.0, 0.0, 2.0];

        // Convert to BF16
        let input_bf16: Vec<half::bf16> = input_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let gamma_bf16: Vec<half::bf16> = gamma_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let beta_bf16: Vec<half::bf16> = beta_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

        let input_storage = device.storage_from_slice(&input_bf16).expect("Failed to create input");
        let gamma_storage = device.storage_from_slice(&gamma_bf16).expect("Failed to create gamma");
        let beta_storage = device.storage_from_slice(&beta_bf16).expect("Failed to create beta");

        let output_storage = layer_norm_bf16_gpu(
            &device, &input_storage, &gamma_storage, &beta_storage,
            batch_size, hidden_size, eps
        ).expect("bf16 layer_norm failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        let expected = cpu_layer_norm(&input_f32, &gamma_f32, &beta_f32, batch_size, hidden_size, eps);

        println!("GPU BF16 output: {:?}", output);
        println!("CPU F32 expected: {:?}", expected);

        for i in 0..output.len() {
            let error = (output[i] - expected[i]).abs();
            assert!(
                error < 1e-2,
                "Mismatch at index {}: got {}, expected {}, error {}",
                i, output[i], expected[i], error
            );
        }

        println!("✅ BF16 layer norm with scale/shift test passed!");
    }

    #[test]
    fn test_bf16_layer_norm_large() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing large BF16 layer norm on: {}", device.adapter_info().name);

        // Typical transformer dimensions
        let batch_size = 32;
        let hidden_size = 768;
        let eps = 1e-5;

        // Random-ish data in F32
        let input_f32: Vec<f32> = (0..batch_size * hidden_size)
            .map(|i| ((i as f32 * 0.1234).sin() + (i as f32 * 0.5678).cos()) * 2.0)
            .collect();

        let gamma_f32: Vec<f32> = (0..hidden_size).map(|i| 0.5 + (i as f32 / hidden_size as f32)).collect();
        let beta_f32: Vec<f32> = (0..hidden_size).map(|i| (i as f32 / hidden_size as f32) - 0.5).collect();

        // Convert to BF16
        let input_bf16: Vec<half::bf16> = input_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let gamma_bf16: Vec<half::bf16> = gamma_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let beta_bf16: Vec<half::bf16> = beta_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

        let input_storage = device.storage_from_slice(&input_bf16).expect("Failed to create input");
        let gamma_storage = device.storage_from_slice(&gamma_bf16).expect("Failed to create gamma");
        let beta_storage = device.storage_from_slice(&beta_bf16).expect("Failed to create beta");

        let output_storage = layer_norm_bf16_gpu(
            &device, &input_storage, &gamma_storage, &beta_storage,
            batch_size, hidden_size, eps
        ).expect("bf16 layer_norm failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        let expected = cpu_layer_norm(&input_f32, &gamma_f32, &beta_f32, batch_size, hidden_size, eps);

        // Check max error
        let mut max_error = 0.0f32;
        for i in 0..output.len() {
            max_error = max_error.max((output[i] - expected[i]).abs());
        }
        println!("Max error: {:.6e}", max_error);
        assert!(max_error < 0.1, "Max error too large: {}", max_error);

        println!("✅ Large BF16 layer norm test passed! ({} × {})", batch_size, hidden_size);
    }
}
