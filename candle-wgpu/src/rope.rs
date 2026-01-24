//! Rotary Positional Encoding (RoPE) for wgpu backend
//!
//! RoPE is used by modern LLMs like Qwen, LLaMA, etc. to encode positional information
//! into queries and keys without adding parameters.
//!
//! The transformation applies:
//!   q'[2i]   = q[2i] * cos(θ) - q[2i+1] * sin(θ)
//!   q'[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)
//!
//! Where θ depends on position and dimension.

use crate::device::{ShaderType, WgpuDevice};
use crate::storage::WgpuStorage;
use candle_core::backend::BackendStorage;
use candle_core::DType;
use std::sync::Arc;

/// Parameters for RoPE
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct RopeParams {
    /// Sequence length
    pub seq_len: u32,
    /// Head dimension (must be even)
    pub head_dim: u32,
    /// Number of attention heads
    pub num_heads: u32,
    /// Padding for alignment
    pub _padding: u32,
}

/// Precompute cos/sin caches for RoPE
///
/// # Arguments
/// * `seq_len` - Maximum sequence length
/// * `head_dim` - Dimension of each attention head (must be even)
/// * `base` - Base for the positional encoding (default: 10000.0)
///
/// # Returns
/// (cos_cache, sin_cache) each of shape [seq_len, head_dim/2]
pub fn compute_rope_cache(seq_len: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim % 2 == 0, "head_dim must be even");

    let half_dim = head_dim / 2;
    let mut cos_cache = vec![0.0f32; seq_len * half_dim];
    let mut sin_cache = vec![0.0f32; seq_len * half_dim];

    for pos in 0..seq_len {
        for i in 0..half_dim {
            // θ_i = pos / base^(2i/d)
            let theta = (pos as f32) / base.powf((2 * i) as f32 / head_dim as f32);
            let idx = pos * half_dim + i;
            cos_cache[idx] = theta.cos();
            sin_cache[idx] = theta.sin();
        }
    }

    (cos_cache, sin_cache)
}

/// Apply RoPE to query/key tensor in-place on GPU
///
/// # Arguments
/// * `device` - The wgpu device
/// * `q` - Query/key tensor [seq_len, num_heads, head_dim] (modified in-place)
/// * `cos_cache` - Precomputed cosine values [seq_len, head_dim/2]
/// * `sin_cache` - Precomputed sine values [seq_len, head_dim/2]
/// * `seq_len` - Sequence length
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Head dimension
pub fn apply_rope_gpu(
    device: &WgpuDevice,
    q: &WgpuStorage,
    cos_cache: &WgpuStorage,
    sin_cache: &WgpuStorage,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> candle_core::Result<WgpuStorage> {
    assert_eq!(q.dtype(), DType::F32, "rope only supports F32");
    assert!(head_dim % 2 == 0, "head_dim must be even");
    assert_eq!(q.count(), seq_len * num_heads * head_dim);
    assert_eq!(cos_cache.count(), seq_len * head_dim / 2);
    assert_eq!(sin_cache.count(), seq_len * head_dim / 2);

    let output_size = seq_len * num_heads * head_dim;
    let output_bytes = output_size * std::mem::size_of::<f32>();

    // Create output buffer with copy of input
    let output_buffer = device.create_buffer(
        output_bytes as u64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        "rope_output",
    );

    // Copy input to output first (RoPE modifies in-place)
    {
        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rope_copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(q.buffer(), 0, &output_buffer, 0, output_bytes as u64);
        device.queue().submit(std::iter::once(encoder.finish()));
    }

    // Create params buffer
    let params = RopeParams {
        seq_len: seq_len as u32,
        head_dim: head_dim as u32,
        num_heads: num_heads as u32,
        _padding: 0,
    };
    let params_bytes = bytemuck::bytes_of(&params);
    let params_buffer = device.create_buffer_init(
        params_bytes,
        wgpu::BufferUsages::UNIFORM,
        "rope_params",
    );

    // Execute RoPE shader
    device.with_pipeline(ShaderType::RopeF32, |cached| {
        let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rope_bind_group"),
            layout: &cached.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cos_cache.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: sin_cache.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rope_encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("rope_pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&cached.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch: [seq_len, num_heads, head_dim/2]
            // Shader workgroup_size is (64, 1, 1), so we dispatch (ceil(seq_len/64), num_heads, head_dim/2)
            let workgroups_x = (seq_len as u32 + 63) / 64;
            pass.dispatch_workgroups(workgroups_x, num_heads as u32, (head_dim / 2) as u32);
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

/// CPU reference implementation for testing
#[cfg(test)]
fn cpu_apply_rope(
    q: &[f32],
    cos_cache: &[f32],
    sin_cache: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let half_dim = head_dim / 2;
    let mut output = q.to_vec();

    for pos in 0..seq_len {
        for head in 0..num_heads {
            for i in 0..half_dim {
                let base_idx = pos * num_heads * head_dim + head * head_dim;
                let idx0 = base_idx + i * 2;
                let idx1 = idx0 + 1;

                let cos_idx = pos * half_dim + i;
                let cos_val = cos_cache[cos_idx];
                let sin_val = sin_cache[cos_idx];

                let q0 = q[idx0];
                let q1 = q[idx1];

                output[idx0] = q0 * cos_val - q1 * sin_val;
                output[idx1] = q0 * sin_val + q1 * cos_val;
            }
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
    fn test_rope_cache() {
        let seq_len = 4;
        let head_dim = 4;
        let base = 10000.0;

        let (cos_cache, sin_cache) = compute_rope_cache(seq_len, head_dim, base);

        // Position 0 should have all cos=1, sin=0
        assert!((cos_cache[0] - 1.0).abs() < 1e-5);
        assert!((sin_cache[0]).abs() < 1e-5);
        assert!((cos_cache[1] - 1.0).abs() < 1e-5);
        assert!((sin_cache[1]).abs() < 1e-5);

        // Later positions should have varying values
        println!("cos_cache: {:?}", cos_cache);
        println!("sin_cache: {:?}", sin_cache);
    }

    #[test]
    fn test_rope_simple() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        let info = device.adapter_info();
        println!("Testing RoPE on: {} ({:?})", info.name, info.backend);

        let seq_len = 2;
        let num_heads = 2;
        let head_dim = 4;
        let base = 10000.0;

        // Create input: [seq_len, num_heads, head_dim] = [2, 2, 4]
        // Flattened: seq_len * num_heads * head_dim = 16 elements
        let q_data: Vec<f32> = vec![
            // pos=0, head=0
            1.0, 0.0, 0.5, 0.5,
            // pos=0, head=1
            0.0, 1.0, -0.5, 0.5,
            // pos=1, head=0
            1.0, 1.0, 0.0, 0.0,
            // pos=1, head=1
            2.0, 0.0, 1.0, -1.0,
        ];

        let (cos_cache, sin_cache) = compute_rope_cache(seq_len, head_dim, base);

        // Create storages
        let q_storage = device.storage_from_slice(&q_data).expect("Failed to create q");
        let cos_storage = device.storage_from_slice(&cos_cache).expect("Failed to create cos");
        let sin_storage = device.storage_from_slice(&sin_cache).expect("Failed to create sin");

        // Apply RoPE on GPU
        let output_storage = apply_rope_gpu(
            &device, &q_storage, &cos_storage, &sin_storage,
            seq_len, num_heads, head_dim
        ).expect("rope failed");

        // Read back result
        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        // Compute expected on CPU
        let expected = cpu_apply_rope(&q_data, &cos_cache, &sin_cache, seq_len, num_heads, head_dim);

        println!("Input: {:?}", q_data);
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

        // Position 0 should be unchanged (cos=1, sin=0)
        // But with small numerical errors from float operations
        println!("✅ RoPE test passed!");
    }

    #[test]
    fn test_rope_large() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing large RoPE on: {}", device.adapter_info().name);

        // Typical transformer dimensions
        let seq_len = 128;
        let num_heads = 8;
        let head_dim = 64;
        let base = 10000.0;

        // Random-ish input
        let q_data: Vec<f32> = (0..seq_len * num_heads * head_dim)
            .map(|i| (i as f32 * 0.1234).sin())
            .collect();

        let (cos_cache, sin_cache) = compute_rope_cache(seq_len, head_dim, base);

        let q_storage = device.storage_from_slice(&q_data).expect("Failed to create q");
        let cos_storage = device.storage_from_slice(&cos_cache).expect("Failed to create cos");
        let sin_storage = device.storage_from_slice(&sin_cache).expect("Failed to create sin");

        let output_storage = apply_rope_gpu(
            &device, &q_storage, &cos_storage, &sin_storage,
            seq_len, num_heads, head_dim
        ).expect("rope failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        let expected = cpu_apply_rope(&q_data, &cos_cache, &sin_cache, seq_len, num_heads, head_dim);

        // Check max error
        let mut max_error = 0.0f32;
        for i in 0..output.len() {
            max_error = max_error.max((output[i] - expected[i]).abs());
        }
        println!("Max error: {:.6e}", max_error);
        assert!(max_error < 1e-4, "Max error too large: {}", max_error);

        // Verify rotation property: magnitude should be preserved
        // |q'| = |q| for each pair
        let half_dim = head_dim / 2;
        for pos in 0..seq_len {
            for head in 0..num_heads {
                for i in 0..half_dim {
                    let base_idx = pos * num_heads * head_dim + head * head_dim;
                    let idx0 = base_idx + i * 2;
                    let idx1 = idx0 + 1;

                    let orig_mag = (q_data[idx0].powi(2) + q_data[idx1].powi(2)).sqrt();
                    let new_mag = (output[idx0].powi(2) + output[idx1].powi(2)).sqrt();

                    assert!(
                        (orig_mag - new_mag).abs() < 1e-4,
                        "Magnitude not preserved at pos={}, head={}, i={}",
                        pos, head, i
                    );
                }
            }
        }

        println!("✅ Large RoPE test passed! ({} × {} × {})", seq_len, num_heads, head_dim);
    }

    #[test]
    fn test_rope_identity_at_pos_zero() {
        if !is_available() {
            println!("Skipping test: no wgpu adapter available");
            return;
        }

        let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
        println!("Testing RoPE identity at position 0 on: {}", device.adapter_info().name);

        // At position 0, rotation angle should be 0, so output = input
        let seq_len = 1;
        let num_heads = 2;
        let head_dim = 8;
        let base = 10000.0;

        let q_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,  // head 0
            -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, // head 1
        ];

        let (cos_cache, sin_cache) = compute_rope_cache(seq_len, head_dim, base);

        let q_storage = device.storage_from_slice(&q_data).expect("Failed to create q");
        let cos_storage = device.storage_from_slice(&cos_cache).expect("Failed to create cos");
        let sin_storage = device.storage_from_slice(&sin_cache).expect("Failed to create sin");

        let output_storage = apply_rope_gpu(
            &device, &q_storage, &cos_storage, &sin_storage,
            seq_len, num_heads, head_dim
        ).expect("rope failed");

        let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
        let output: Vec<f32> = match output_cpu {
            candle_core::CpuStorage::F32(data) => data,
            _ => panic!("Unexpected dtype"),
        };

        println!("Input: {:?}", q_data);
        println!("Output: {:?}", output);

        // At position 0, output should equal input
        for i in 0..output.len() {
            let error = (output[i] - q_data[i]).abs();
            assert!(
                error < 1e-5,
                "Position 0 should be identity: got {}, expected {}, error {}",
                output[i], q_data[i], error
            );
        }

        println!("✅ RoPE identity test passed!");
    }
}
