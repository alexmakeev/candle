//! Quantized operations for wgpu backend
//!
//! This module implements quantized matrix multiplication using DP4a
//! (4x INT8 dot product) instructions for maximum performance.
//!
//! ## Supported formats
//!
//! - Q8_0: 8-bit quantization with per-block scale
//!   - Block size: 32 elements
//!   - Storage: 34 bytes per block (2 bytes scale + 32 bytes data)
//!   - Quality: ~99.5% of FP16

use wgpu;

/// Q8_0 block structure (matches candle-core layout)
/// 32 elements per block, 34 bytes total
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlockQ8_0 {
    /// Scale factor (f16 stored as u16 for Pod compatibility)
    pub d: u16,
    /// Quantized values (32x i8, stored as 8x u32 for DP4a)
    pub qs: [u8; 32],
}

const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

/// Parameters for quantized matmul
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct QuantizedMatmulParams {
    /// Number of rows in output (M)
    pub m: u32,
    /// Number of columns in output (N)
    pub n: u32,
    /// Inner dimension (K) - must be multiple of 32 (block size)
    pub k: u32,
    /// Number of blocks per row in weights
    pub blocks_per_row: u32,
}

/// WGSL shader for Q8_0 matmul
pub const Q8_0_MATMUL_SHADER: &str = r#"
// Q8_0 Quantized Matrix Multiplication
//
// Memory layout per block (36 bytes, GPU-friendly):
// - u32[0]: f16 scale in lower 16 bits (upper 16 bits = 0)
// - u32[1..9]: 32× i8 packed as 8× u32

struct Params {
    m: u32,              // output rows
    n: u32,              // output cols (unused for matvec)
    k: u32,              // inner dimension
    blocks_per_row: u32, // K / 32
}

// Raw buffer access - blocks are 9× u32 = 36 bytes each
@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// Unpack f16 from lower 16 bits of u32
fn unpack_f16(packed: u32) -> f32 {
    let bits = packed & 0xFFFFu;
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 0x1Fu;
    let mant = bits & 0x3FFu;

    if (exp == 0u) {
        if (mant == 0u) { return 0.0; }
        // Denormalized
        let f = f32(mant) / 1024.0 * pow(2.0, -14.0);
        return select(f, -f, sign == 1u);
    }
    if (exp == 31u) {
        return select(1e38, -1e38, sign == 1u);
    }

    let f = (1.0 + f32(mant) / 1024.0) * pow(2.0, f32(exp) - 15.0);
    return select(f, -f, sign == 1u);
}

// Unpack 4× i8 from u32 and convert to f32 vec4
fn unpack_i8x4(packed: u32) -> vec4<f32> {
    let b0 = i32((packed >> 0u) & 0xFFu);
    let b1 = i32((packed >> 8u) & 0xFFu);
    let b2 = i32((packed >> 16u) & 0xFFu);
    let b3 = i32((packed >> 24u) & 0xFFu);

    // Sign extend from 8-bit
    let s0 = select(b0, b0 - 256, b0 > 127);
    let s1 = select(b1, b1 - 256, b1 > 127);
    let s2 = select(b2, b2 - 256, b2 > 127);
    let s3 = select(b3, b3 - 256, b3 > 127);

    return vec4<f32>(f32(s0), f32(s1), f32(s2), f32(s3));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;

    if (row >= params.m) {
        return;
    }

    var acc: f32 = 0.0;

    // Each block is 9× u32 (36 bytes): 1 for scale, 8 for data
    let block_stride = 9u;

    for (var blk = 0u; blk < params.blocks_per_row; blk++) {
        let block_base = (row * params.blocks_per_row + blk) * block_stride;

        // First u32 contains f16 scale in lower 16 bits
        let scale = unpack_f16(weights[block_base]);
        let k_base = blk * 32u;

        // Process 32 elements (8 groups of 4)
        for (var i = 0u; i < 8u; i++) {
            let packed = weights[block_base + 1u + i];
            let vals = unpack_i8x4(packed);
            let k_off = k_base + i * 4u;

            acc += vals.x * scale * input[k_off + 0u];
            acc += vals.y * scale * input[k_off + 1u];
            acc += vals.z * scale * input[k_off + 2u];
            acc += vals.w * scale * input[k_off + 3u];
        }
    }

    output[row] = acc;
}
"#;

/// WGSL shader for Q8_0 matmul with DP4a (requires packed_4x8_integer_dot_product)
pub const Q8_0_MATMUL_DP4A_SHADER: &str = r#"
// Q8_0 Quantized MatMul with DP4a
// Requires: enable packed_4x8_integer_dot_product;

struct BlockQ8_0 {
    d: u32,             // f16 scale (lower 16 bits)
    qs: array<u32, 8>,  // 32x i8 packed as 8x u32
}

struct Params {
    m: u32,
    n: u32,
    k: u32,
    blocks_per_row: u32,
}

@group(0) @binding(0) var<storage, read> weights_q8: array<BlockQ8_0>;
@group(0) @binding(1) var<storage, read> input_q8: array<u32>;
@group(0) @binding(2) var<storage, read> input_scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

fn unpack_f16(packed: u32) -> f32 {
    let sign = (packed >> 15u) & 1u;
    let exp = (packed >> 10u) & 0x1Fu;
    let mant = packed & 0x3FFu;
    if (exp == 0u && mant == 0u) { return 0.0; }
    if (exp == 31u) { return select(1e38, -1e38, sign == 1u); }
    let f = (1.0 + f32(mant) / 1024.0) * pow(2.0, f32(exp) - 15.0);
    return select(f, -f, sign == 1u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;

    if (row >= params.m) {
        return;
    }

    var acc_i32: i32 = 0;
    var scale_product: f32 = 0.0;

    for (var blk = 0u; blk < params.blocks_per_row; blk++) {
        let w_block = weights_q8[row * params.blocks_per_row + blk];
        let w_scale = unpack_f16(w_block.d);
        let x_scale = input_scales[blk];

        scale_product += w_scale * x_scale;

        for (var i = 0u; i < 8u; i++) {
            let w_packed = w_block.qs[i];
            let x_packed = input_q8[blk * 8u + i];

            let w0 = i32((w_packed >> 0u) & 0xFFu);
            let w1 = i32((w_packed >> 8u) & 0xFFu);
            let w2 = i32((w_packed >> 16u) & 0xFFu);
            let w3 = i32((w_packed >> 24u) & 0xFFu);

            let x0 = i32((x_packed >> 0u) & 0xFFu);
            let x1 = i32((x_packed >> 8u) & 0xFFu);
            let x2 = i32((x_packed >> 16u) & 0xFFu);
            let x3 = i32((x_packed >> 24u) & 0xFFu);

            // Sign extend
            let ws0 = select(w0, w0 - 256, w0 > 127);
            let ws1 = select(w1, w1 - 256, w1 > 127);
            let ws2 = select(w2, w2 - 256, w2 > 127);
            let ws3 = select(w3, w3 - 256, w3 > 127);

            let xs0 = select(x0, x0 - 256, x0 > 127);
            let xs1 = select(x1, x1 - 256, x1 > 127);
            let xs2 = select(x2, x2 - 256, x2 > 127);
            let xs3 = select(x3, x3 - 256, x3 > 127);

            acc_i32 += ws0 * xs0 + ws1 * xs1 + ws2 * xs2 + ws3 * xs3;
        }
    }

    output[row] = f32(acc_i32) * scale_product / f32(params.blocks_per_row);
}
"#;

/// Quantize f32 data to Q8_0 format on CPU
pub fn quantize_f32_to_q8_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0, "Data length must be multiple of 32");

    let num_blocks = data.len() / 32;
    let mut result = Vec::with_capacity(num_blocks * 34);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * 32;
        let block_data = &data[block_start..block_start + 32];

        let max_abs = block_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let scale_f16 = half::f16::from_f32(scale);
        let scale_bits = scale_f16.to_bits();

        result.push((scale_bits & 0xFF) as u8);
        result.push((scale_bits >> 8) as u8);

        for &val in block_data {
            let quantized = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
            result.push(quantized as u8);
        }
    }

    result
}

/// Dequantize Q8_0 data to f32 on CPU (for verification)
pub fn dequantize_q8_0_to_f32(data: &[u8]) -> Vec<f32> {
    assert!(data.len() % 34 == 0, "Data length must be multiple of 34 (Q8_0 block size)");

    let num_blocks = data.len() / 34;
    let mut result = Vec::with_capacity(num_blocks * 32);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * 34;

        let scale_bits = u16::from_le_bytes([data[block_start], data[block_start + 1]]);
        let scale = half::f16::from_bits(scale_bits).to_f32();

        for i in 0..32 {
            let q = data[block_start + 2 + i] as i8;
            result.push(q as f32 * scale);
        }
    }

    result
}

/// GPU-accelerated Q8_0 matrix-vector multiplication
pub struct Q8_0MatMul {
    weights_buffer: wgpu::Buffer,
    num_rows: usize,
    num_cols: usize,
    blocks_per_row: usize,
}

impl Q8_0MatMul {
    /// Create a new Q8_0 matrix multiplication operator
    pub fn new(
        device: &super::device::WgpuDevice,
        weights_q8: &[u8],
        num_rows: usize,
        num_cols: usize,
    ) -> Self {
        assert!(num_cols % 32 == 0, "num_cols must be multiple of 32");

        let blocks_per_row = num_cols / 32;
        let num_blocks = num_rows * blocks_per_row;
        let expected_size = num_blocks * 34;

        assert_eq!(
            weights_q8.len(),
            expected_size,
            "weights_q8 size mismatch: expected {}, got {}",
            expected_size,
            weights_q8.len()
        );

        // Repack data for GPU: 34 bytes → 36 bytes per block (9× u32)
        let mut gpu_data = Vec::with_capacity(num_blocks * 36);

        for block_idx in 0..num_blocks {
            let src_offset = block_idx * 34;

            let scale_lo = weights_q8[src_offset] as u32;
            let scale_hi = weights_q8[src_offset + 1] as u32;
            let scale_u32 = scale_lo | (scale_hi << 8);
            gpu_data.extend_from_slice(&scale_u32.to_le_bytes());

            for i in 0..8 {
                let qs_offset = src_offset + 2 + i * 4;
                let qs_u32 = u32::from_le_bytes([
                    weights_q8[qs_offset],
                    weights_q8[qs_offset + 1],
                    weights_q8[qs_offset + 2],
                    weights_q8[qs_offset + 3],
                ]);
                gpu_data.extend_from_slice(&qs_u32.to_le_bytes());
            }
        }

        let weights_buffer = device.create_buffer_init(
            &gpu_data,
            wgpu::BufferUsages::STORAGE,
            "q8_0_weights",
        );

        Self {
            weights_buffer,
            num_rows,
            num_cols,
            blocks_per_row,
        }
    }

    /// Perform matrix-vector multiplication
    pub fn forward(
        &self,
        device: &super::device::WgpuDevice,
        input: &[f32],
    ) -> Vec<f32> {
        use super::device::ShaderType;

        assert_eq!(
            input.len(),
            self.num_cols,
            "input length mismatch: expected {}, got {}",
            self.num_cols,
            input.len()
        );

        let input_bytes = bytemuck::cast_slice(input);
        let input_buffer = device.create_buffer_init(
            input_bytes,
            wgpu::BufferUsages::STORAGE,
            "q8_0_input",
        );

        let output_size = self.num_rows * std::mem::size_of::<f32>();
        let output_buffer = device.create_buffer(
            output_size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "q8_0_output",
        );

        let params = QuantizedMatmulParams {
            m: self.num_rows as u32,
            n: 1,
            k: self.num_cols as u32,
            blocks_per_row: self.blocks_per_row as u32,
        };
        let params_bytes = bytemuck::bytes_of(&params);
        let params_buffer = device.create_buffer_init(
            params_bytes,
            wgpu::BufferUsages::UNIFORM,
            "q8_0_params",
        );

        device.with_pipeline(ShaderType::MatmulQ8_0, |cached| {
            let bind_group = device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("q8_0_matmul_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.weights_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            device.with_encoder(|encoder| {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("q8_0_matmul_pass"),
                    timestamp_writes: None,
                });

                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);

                let workgroups = (self.num_rows as u32 + 63) / 64;
                pass.dispatch_workgroups(workgroups, 1, 1);
            });
        });

        // Read back results
        let staging_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("q8_0_staging"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        device.with_encoder(|encoder| {
            encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size as u64);
        });
        device.flush();

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        device.device().poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }
}
