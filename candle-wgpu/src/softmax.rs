//! Softmax operations for wgpu backend
//!
//! This module provides efficient fused softmax implementation on GPU.
//! The fused implementation performs max/subtract/exp/sum/divide in a single kernel,
//! avoiding multiple GPU round trips.

use crate::device::{ShaderType, WgpuDevice};
use crate::storage::WgpuStorage;
use candle_core::backend::BackendStorage;
use candle_core::DType;
use std::sync::Arc;

/// Parameters for softmax/reduce operations
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SoftmaxParams {
    /// Number of rows (batch dimension)
    pub num_rows: u32,
    /// Size of each row (dimension to reduce over)
    pub row_size: u32,
}

/// GPU-accelerated softmax along the last dimension
///
/// Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
/// for each row independently.
///
/// # Arguments
/// * `device` - The wgpu device
/// * `input` - Input tensor data (flattened, row-major)
/// * `num_rows` - Number of rows
/// * `row_size` - Size of each row (the dimension to softmax over)
///
/// # Returns
/// Output tensor data with same shape as input
pub fn softmax_last_dim_gpu(
    device: &WgpuDevice,
    input: &WgpuStorage,
    num_rows: usize,
    row_size: usize,
) -> candle_core::Result<WgpuStorage> {
    assert_eq!(input.dtype(), DType::F32, "softmax_last_dim_gpu only supports F32");
    assert_eq!(input.count(), num_rows * row_size);

    let output_size = num_rows * row_size;
    let output_bytes = output_size * std::mem::size_of::<f32>();

    // Create output buffer
    let output_buffer = device.create_buffer(
        output_bytes as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        "softmax_output",
    );

    // Create params buffer
    let params = SoftmaxParams {
        num_rows: num_rows as u32,
        row_size: row_size as u32,
    };
    let params_bytes = bytemuck::bytes_of(&params);
    let params_buffer = device.create_buffer_init(
        params_bytes,
        wgpu::BufferUsages::UNIFORM,
        "softmax_params",
    );

    // Execute fused softmax shader
    device.with_pipeline(ShaderType::SoftmaxFused, |cached| {
        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_fused_bind_group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("softmax_fused_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_fused_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // One workgroup per row
            pass.dispatch_workgroups(num_rows as u32, 1, 1);
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

/// GPU-accelerated reduce max along the last dimension
///
/// # Arguments
/// * `device` - The wgpu device
/// * `input` - Input tensor data
/// * `num_rows` - Number of rows
/// * `row_size` - Size of each row
///
/// # Returns
/// Output tensor with shape [num_rows]
pub fn reduce_max_last_dim_gpu(
    device: &WgpuDevice,
    input: &WgpuStorage,
    num_rows: usize,
    row_size: usize,
) -> candle_core::Result<WgpuStorage> {
    assert_eq!(input.dtype(), DType::F32);
    assert_eq!(input.count(), num_rows * row_size);

    let output_bytes = num_rows * std::mem::size_of::<f32>();

    // Create output buffer
    let output_buffer = device.create_buffer(
        output_bytes as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        "reduce_max_output",
    );

    // Create params buffer
    let params = SoftmaxParams {
        num_rows: num_rows as u32,
        row_size: row_size as u32,
    };
    let params_bytes = bytemuck::bytes_of(&params);
    let params_buffer = device.create_buffer_init(
        params_bytes,
        wgpu::BufferUsages::UNIFORM,
        "reduce_max_params",
    );

    // Execute reduce max shader
    device.with_pipeline(ShaderType::ReduceMaxLastDim, |cached| {
        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce_max_bind_group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reduce_max_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce_max_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // One workgroup per row
            pass.dispatch_workgroups(num_rows as u32, 1, 1);
        }

        device.queue().submit(std::iter::once(encoder.finish()));
    });

    Ok(WgpuStorage::new(
        Arc::new(output_buffer),
        device.clone(),
        num_rows,
        DType::F32,
    ))
}

/// GPU-accelerated reduce sum along the last dimension
pub fn reduce_sum_last_dim_gpu(
    device: &WgpuDevice,
    input: &WgpuStorage,
    num_rows: usize,
    row_size: usize,
) -> candle_core::Result<WgpuStorage> {
    assert_eq!(input.dtype(), DType::F32);
    assert_eq!(input.count(), num_rows * row_size);

    let output_bytes = num_rows * std::mem::size_of::<f32>();

    // Create output buffer
    let output_buffer = device.create_buffer(
        output_bytes as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        "reduce_sum_output",
    );

    // Create params buffer
    let params = SoftmaxParams {
        num_rows: num_rows as u32,
        row_size: row_size as u32,
    };
    let params_bytes = bytemuck::bytes_of(&params);
    let params_buffer = device.create_buffer_init(
        params_bytes,
        wgpu::BufferUsages::UNIFORM,
        "reduce_sum_params",
    );

    // Execute reduce sum shader
    device.with_pipeline(ShaderType::ReduceSumLastDim, |cached| {
        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reduce_sum_bind_group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reduce_sum_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("reduce_sum_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // One workgroup per row
            pass.dispatch_workgroups(num_rows as u32, 1, 1);
        }

        device.queue().submit(std::iter::once(encoder.finish()));
    });

    Ok(WgpuStorage::new(
        Arc::new(output_buffer),
        device.clone(),
        num_rows,
        DType::F32,
    ))
}

/// GPU-accelerated BF16 softmax along the last dimension
///
/// Input: BF16 stored as u16
/// Computation: Convert to F32, compute softmax, return F32 (to be converted to BF16 on CPU)
///
/// # Arguments
/// * `device` - The wgpu device
/// * `input` - Input tensor data (BF16 stored as u16, flattened, row-major)
/// * `num_rows` - Number of rows
/// * `row_size` - Size of each row (the dimension to softmax over)
///
/// # Returns
/// Output tensor data in F32 (caller converts to BF16 if needed)
pub fn softmax_bf16_gpu(
    device: &WgpuDevice,
    input: &WgpuStorage,
    num_rows: usize,
    row_size: usize,
) -> candle_core::Result<WgpuStorage> {
    use candle_core::DType;

    assert_eq!(input.dtype(), DType::BF16, "softmax_bf16_gpu requires BF16 input");
    assert_eq!(input.count(), num_rows * row_size);

    let output_size = num_rows * row_size;
    let output_bytes = output_size * std::mem::size_of::<f32>();

    // Create F32 output buffer
    let output_buffer = device.create_buffer(
        output_bytes as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        "softmax_bf16_output",
    );

    // Create params buffer
    let params = SoftmaxParams {
        num_rows: num_rows as u32,
        row_size: row_size as u32,
    };
    let params_bytes = bytemuck::bytes_of(&params);
    let params_buffer = device.create_buffer_init(
        params_bytes,
        wgpu::BufferUsages::UNIFORM,
        "softmax_bf16_params",
    );

    // Execute BF16 softmax shader
    device.with_pipeline(ShaderType::SoftmaxBF16, |cached| {
        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("softmax_bf16_bind_group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("softmax_bf16_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("softmax_bf16_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // One workgroup per row
            pass.dispatch_workgroups(num_rows as u32, 1, 1);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{is_available, WgpuDevice};
    use candle_core::backend::BackendDevice;

    fn cpu_softmax(input: &[f32], num_rows: usize, row_size: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; input.len()];

        for row in 0..num_rows {
            let offset = row * row_size;
            let row_data = &input[offset..offset + row_size];

            // Find max
            let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            for i in 0..row_size {
                let exp_val = (row_data[i] - max_val).exp();
                output[offset + i] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for i in 0..row_size {
                output[offset + i] /= sum;
            }
        }

        output
    }

    #[test]
    fn test_softmax_fused_simple() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let info = device.adapter_info();
        println!("Testing fused softmax on: {} ({:?})", info.name, info.backend);

        // Test case: 2 rows × 4 elements
        let num_rows = 2;
        let row_size = 4;
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,  // row 0
            2.0, 4.0, 6.0, 8.0,  // row 1
        ];

        // Create input storage
        let input_storage = device.storage_from_slice(&input_data).expect("Failed to create storage");

        // Run GPU softmax
        let output_storage = softmax_last_dim_gpu(&device, &input_storage, num_rows, row_size)
            .expect("softmax failed");

        // Read back result
        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        // Compute expected on CPU
        let expected = cpu_softmax(&input_data, num_rows, row_size);

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

        // Verify softmax properties
        for row in 0..num_rows {
            let offset = row * row_size;
            let row_sum: f32 = output[offset..offset + row_size].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "Row {} sum should be 1.0, got {}",
                row, row_sum
            );

            for i in 0..row_size {
                assert!(
                    output[offset + i] >= 0.0 && output[offset + i] <= 1.0,
                    "Softmax output should be in [0, 1]"
                );
            }
        }

        println!("✅ Fused softmax test passed!");
    }

    #[test]
    fn test_softmax_large() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing large softmax on: {}", device.adapter_info().name);

        // Larger test: 128 rows × 512 elements (typical attention size)
        let num_rows = 128;
        let row_size = 512;
        let input_data: Vec<f32> = (0..num_rows * row_size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        let input_storage = device.storage_from_slice(&input_data).expect("Failed to create storage");

        let output_storage = softmax_last_dim_gpu(&device, &input_storage, num_rows, row_size)
            .expect("softmax failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        let expected = cpu_softmax(&input_data, num_rows, row_size);

        // Check error
        let mut max_error = 0.0f32;
        for i in 0..output.len() {
            max_error = max_error.max((output[i] - expected[i]).abs());
        }
        println!("Max error: {:.6e}", max_error);
        assert!(max_error < 1e-4, "Max error too large: {}", max_error);

        // Check row sums
        for row in 0..num_rows {
            let offset = row * row_size;
            let row_sum: f32 = output[offset..offset + row_size].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "Row {} sum should be 1.0, got {}",
                row, row_sum
            );
        }

        println!("✅ Large softmax test passed! ({} × {})", num_rows, row_size);
    }

    #[test]
    fn test_reduce_max() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing reduce max on: {}", device.adapter_info().name);

        let num_rows = 4;
        let row_size = 8;
        let input_data: Vec<f32> = vec![
            1.0, 5.0, 3.0, 2.0, 4.0, 6.0, 2.0, 1.0,   // max = 6.0
            9.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,   // max = 9.0
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, // max = -1.0
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, // max = 100.0
        ];
        let expected_max: Vec<f32> = vec![6.0, 9.0, -1.0, 100.0];

        let input_storage = device.storage_from_slice(&input_data).expect("Failed to create storage");

        let output_storage = reduce_max_last_dim_gpu(&device, &input_storage, num_rows, row_size)
            .expect("reduce max failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        println!("GPU max: {:?}", output);
        println!("Expected: {:?}", expected_max);

        for i in 0..num_rows {
            assert!(
                (output[i] - expected_max[i]).abs() < 1e-5,
                "Row {} max mismatch: got {}, expected {}",
                i, output[i], expected_max[i]
            );
        }

        println!("✅ Reduce max test passed!");
    }

    #[test]
    fn test_reduce_sum() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing reduce sum on: {}", device.adapter_info().name);

        let num_rows = 3;
        let row_size = 4;
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,   // sum = 10.0
            0.5, 0.5, 0.5, 0.5,   // sum = 2.0
            -1.0, 1.0, -2.0, 2.0, // sum = 0.0
        ];
        let expected_sum: Vec<f32> = vec![10.0, 2.0, 0.0];

        let input_storage = device.storage_from_slice(&input_data).expect("Failed to create storage");

        let output_storage = reduce_sum_last_dim_gpu(&device, &input_storage, num_rows, row_size)
            .expect("reduce sum failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        println!("GPU sum: {:?}", output);
        println!("Expected: {:?}", expected_sum);

        for i in 0..num_rows {
            assert!(
                (output[i] - expected_sum[i]).abs() < 1e-5,
                "Row {} sum mismatch: got {}, expected {}",
                i, output[i], expected_sum[i]
            );
        }

        println!("✅ Reduce sum test passed!");
    }

    #[test]
    fn test_softmax_numerical_stability() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing softmax numerical stability on: {}", device.adapter_info().name);

        // Test with large values that would overflow without max subtraction
        let num_rows = 2;
        let row_size = 4;
        let input_data: Vec<f32> = vec![
            1000.0, 1001.0, 1002.0, 1000.5,  // Large values
            -1000.0, -999.0, -1001.0, -998.0, // Large negative values
        ];

        let input_storage = device.storage_from_slice(&input_data).expect("Failed to create storage");

        let output_storage = softmax_last_dim_gpu(&device, &input_storage, num_rows, row_size)
            .expect("softmax failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        let expected = cpu_softmax(&input_data, num_rows, row_size);

        println!("Input: {:?}", input_data);
        println!("GPU output: {:?}", output);
        println!("CPU expected: {:?}", expected);

        // Should not have NaN or Inf
        for val in &output {
            assert!(!val.is_nan(), "Output contains NaN");
            assert!(!val.is_infinite(), "Output contains Inf");
        }

        // Check against CPU reference
        for i in 0..output.len() {
            let error = (output[i] - expected[i]).abs();
            assert!(
                error < 1e-4,
                "Mismatch at index {}: got {}, expected {}, error {}",
                i, output[i], expected[i], error
            );
        }

        println!("✅ Softmax numerical stability test passed!");
    }

    #[test]
    fn test_bf16_softmax() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let info = device.adapter_info();
        println!("Testing BF16 softmax on: {} ({:?})", info.name, info.backend);

        // Test case: 2 rows × 4 elements
        let num_rows = 2;
        let row_size = 4;
        let input_f32: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,  // row 0
            2.0, 4.0, 6.0, 8.0,  // row 1
        ];

        // Convert to BF16
        let input_bf16: Vec<half::bf16> = input_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

        // Create input storage as BF16
        let input_storage = device.storage_from_slice(&input_bf16).expect("Failed to create storage");

        // Run GPU BF16 softmax (returns F32)
        let output_storage = softmax_bf16_gpu(&device, &input_storage, num_rows, row_size)
            .expect("bf16 softmax failed");

        // Read back result (F32)
        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        // Compute expected on CPU (using F32 for reference)
        let expected = cpu_softmax(&input_f32, num_rows, row_size);

        println!("Input F32: {:?}", input_f32);
        println!("GPU BF16 output: {:?}", output);
        println!("CPU F32 expected: {:?}", expected);

        // Verify results (BF16 has less precision, so allow larger error)
        for i in 0..output.len() {
            let error = (output[i] - expected[i]).abs();
            assert!(
                error < 1e-3,
                "Mismatch at index {}: got {}, expected {}, error {}",
                i, output[i], expected[i], error
            );
        }

        // Verify softmax properties
        for row in 0..num_rows {
            let offset = row * row_size;
            let row_sum: f32 = output[offset..offset + row_size].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "Row {} sum should be 1.0, got {}",
                row, row_sum
            );

            for i in 0..row_size {
                assert!(
                    output[offset + i] >= 0.0 && output[offset + i] <= 1.0,
                    "Softmax output should be in [0, 1]"
                );
            }
        }

        println!("✅ BF16 softmax test passed!");
    }

    #[test]
    fn test_bf16_softmax_large() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing large BF16 softmax on: {}", device.adapter_info().name);

        // Larger test: 64 rows × 256 elements
        let num_rows = 64;
        let row_size = 256;
        let input_f32: Vec<f32> = (0..num_rows * row_size)
            .map(|i| (i as f32 * 0.01).sin())
            .collect();

        // Convert to BF16
        let input_bf16: Vec<half::bf16> = input_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
        let input_storage = device.storage_from_slice(&input_bf16).expect("Failed to create storage");

        let output_storage = softmax_bf16_gpu(&device, &input_storage, num_rows, row_size)
            .expect("bf16 softmax failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        let expected = cpu_softmax(&input_f32, num_rows, row_size);

        // Check error
        let mut max_error = 0.0f32;
        for i in 0..output.len() {
            max_error = max_error.max((output[i] - expected[i]).abs());
        }
        println!("Max error: {:.6e}", max_error);
        assert!(max_error < 1e-2, "Max error too large: {}", max_error);

        // Check row sums
        for row in 0..num_rows {
            let offset = row * row_size;
            let row_sum: f32 = output[offset..offset + row_size].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-3,
                "Row {} sum should be 1.0, got {}",
                row, row_sum
            );
        }

        println!("✅ Large BF16 softmax test passed! ({} × {})", num_rows, row_size);
    }
}
