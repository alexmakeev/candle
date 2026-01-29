//! wgpu storage implementation

use super::device::{ShaderType, WgpuDevice};
use super::error::WgpuError;
use super::flags::shader_flags;
use crate::backend::{BackendDevice, BackendStorage};
use crate::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result};
use std::sync::Arc;
use wgpu::Buffer;

/// Dimensions for F32 matmul shader (must match WGSL struct: M, N, K)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulDimensions {
    m: u32,
    n: u32,
    k: u32,
    _padding: u32,
}

/// Parameters for strided copy shader (must match WGSL struct, up to 6 dims)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CopyStridedParams {
    ndims: u32,
    src_offset: u32,
    dst_offset: u32,
    elem_count: u32,
    // Shape (up to 6 dims, unused dims = 1)
    shape: [u32; 6],
    // Source strides (up to 6 dims, unused = 0)
    strides: [u32; 6],
}

/// Dimensions for batched BF16 matmul shader (must match WGSL struct)
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulBF16Dimensions {
    m: u32,
    n: u32,
    k: u32,
    batch_count: u32,
    a_batch_stride: u32,
    b_batch_stride: u32,
    c_batch_stride: u32,
    _padding: u32,
}

/// Storage for tensors on a wgpu device
#[derive(Debug, Clone)]
pub struct WgpuStorage {
    buffer: Arc<Buffer>,
    device: WgpuDevice,
    count: usize,
    dtype: DType,
}

impl WgpuStorage {
    pub fn new(buffer: Arc<Buffer>, device: WgpuDevice, count: usize, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            count,
            dtype,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn count(&self) -> usize {
        self.count
    }

    fn to_cpu<T: bytemuck::Pod>(&self) -> Result<Vec<T>> {
        let size = self.count * std::mem::size_of::<T>();
        // wgpu requires COPY_BUFFER_ALIGNMENT (4 bytes) for buffer copies
        const COPY_ALIGNMENT: u64 = wgpu::COPY_BUFFER_ALIGNMENT;
        let aligned_size = ((size as u64 + COPY_ALIGNMENT - 1) / COPY_ALIGNMENT) * COPY_ALIGNMENT;

        let staging_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: aligned_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, aligned_size);
        self.device.queue().submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.device().poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| crate::Error::Msg(WgpuError::BufferMapFailed.to_string()))?
            .map_err(|_| crate::Error::Msg(WgpuError::BufferMapFailed.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        // Only copy the actual data size, not the aligned padding
        let result: Vec<T> = bytemuck::cast_slice(&data[..size]).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    fn from_cpu_op<F>(&self, layout: &Layout, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage, &Layout) -> Result<CpuStorage>,
    {
        let cpu_storage = self.to_cpu_storage()?;
        let result = f(&cpu_storage, layout)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn from_cpu_binary_op<F>(
        &self,
        rhs: &Self,
        lhs_layout: &Layout,
        rhs_layout: &Layout,
        f: F,
    ) -> Result<Self>
    where
        F: FnOnce(&CpuStorage, &CpuStorage, &Layout, &Layout) -> Result<CpuStorage>,
    {
        let lhs_cpu = self.to_cpu_storage()?;
        let rhs_cpu = rhs.to_cpu_storage()?;
        let result = f(&lhs_cpu, &rhs_cpu, lhs_layout, rhs_layout)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    /// BF16 matmul on GPU. Supports batched (b >= 1) and contiguous inputs.
    /// A[b, m, k] @ B[b, k, n] → C[b, m, n] in BF16.
    /// Internally accumulates in F32, batch dimension dispatched via global_id.z.
    fn matmul_bf16_gpu(&self, rhs: &Self, b: usize, m: usize, n: usize, k: usize) -> Result<Self> {
        let total_output_size = b * m * n;
        let total_output_f32_bytes = total_output_size * std::mem::size_of::<f32>();

        let output_f32_buffer = self.device.create_buffer(
            total_output_f32_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "matmul_bf16_output_f32",
        );

        let dims = MatmulBF16Dimensions {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            batch_count: b as u32,
            a_batch_stride: (m * k) as u32,
            b_batch_stride: (k * n) as u32,
            c_batch_stride: (m * n) as u32,
            _padding: 0,
        };
        let dims_bytes = bytemuck::bytes_of(&dims);
        let dims_buffer = self.device.create_buffer_init(
            dims_bytes,
            wgpu::BufferUsages::UNIFORM,
            "matmul_bf16_dims",
        );

        let workgroups_x = (m as u32 + 15) / 16;
        let workgroups_y = (n as u32 + 15) / 16;
        let workgroups_z = b as u32;

        self.device.with_pipeline(ShaderType::MatmulBF16, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_bf16_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_f32_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_bf16_encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("matmul_bf16_pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&cached.pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        // Convert F32 output to BF16 on GPU (no CPU readback)
        let bf16_output = self.cast_f32_to_bf16_gpu(&output_f32_buffer, total_output_size)?;

        Ok(bf16_output)
    }

    /// BF16 RMS norm on GPU.
    /// x: [num_rows, hidden_size] BF16, alpha: [hidden_size] BF16
    /// output: [num_rows, hidden_size] BF16
    pub fn rms_norm_bf16_gpu(
        &self,
        alpha: &Self,
        x_layout: &Layout,
        _alpha_layout: &Layout,
        eps: f32,
    ) -> Result<Self> {
        let shape = x_layout.shape();
        let dims = shape.dims();
        let hidden_size = *dims.last().ok_or_else(|| {
            crate::Error::Msg("rms_norm: empty shape".to_string())
        })?;
        let num_rows = shape.elem_count() / hidden_size;

        let output_bytes = shape.elem_count() * 2; // BF16
        let output_buffer = self.device.create_buffer(
            output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "rms_norm_bf16_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct RmsNormParams {
            num_rows: u32,
            hidden_size: u32,
            eps: f32,
            input_offset: u32,
        }

        let params = RmsNormParams {
            num_rows: num_rows as u32,
            hidden_size: hidden_size as u32,
            eps,
            input_offset: x_layout.start_offset() as u32,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "rms_norm_bf16_params",
        );

        self.device.with_pipeline(ShaderType::RmsNormBF16, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rms_norm_bf16_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: alpha.buffer.as_entire_binding(),
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

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rms_norm_bf16_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rms_norm_bf16_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // One workgroup per row
                pass.dispatch_workgroups(num_rows as u32, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            shape.elem_count(),
            DType::BF16,
        ))
    }

    /// BF16 softmax (last dim) on GPU.
    /// x: [..., last_dim] BF16 contiguous
    /// output: same shape, BF16
    pub fn softmax_bf16_gpu(&self, layout: &Layout) -> Result<Self> {
        let shape = layout.shape();
        let dims = shape.dims();
        let last_dim = *dims.last().ok_or_else(|| {
            crate::Error::Msg("softmax: empty shape".to_string())
        })?;
        let num_rows = shape.elem_count() / last_dim;

        // Softmax BF16 shader outputs F32, we need to cast back
        let f32_output_bytes = shape.elem_count() * std::mem::size_of::<f32>();
        let f32_output_buffer = self.device.create_buffer(
            f32_output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "softmax_bf16_f32_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct SoftmaxParams {
            num_rows: u32,
            row_size: u32,
        }

        let params = SoftmaxParams {
            num_rows: num_rows as u32,
            row_size: last_dim as u32,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "softmax_bf16_params",
        );

        self.device.with_pipeline(ShaderType::SoftmaxBF16, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("softmax_bf16_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: f32_output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
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

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        // Cast F32 output back to BF16 on GPU
        self.cast_f32_to_bf16_gpu(&f32_output_buffer, shape.elem_count())
    }

    /// BF16 RoPE on GPU. Returns BF16 output.
    /// src: [b*h, t*d] BF16 (flat), cos/sin: [t, d/2] or [b*t, d/2] BF16
    pub fn rope_bf16_gpu(
        &self,
        cos: &Self,
        sin: &Self,
        l_src: &Layout,
        l_cos: &Layout,
        l_sin: &Layout,
    ) -> Result<Self> {
        let (b, h, t, d) = l_src.shape().dims4()?;
        let bh = b * h;
        let td = t * d;
        let el = bh * td;

        let stride_b: u32 = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            (h * t * d) as u32
        } else {
            0u32
        };

        // F32 intermediate output
        let f32_output_bytes = el * std::mem::size_of::<f32>();
        let f32_output_buffer = self.device.create_buffer(
            f32_output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "rope_bf16_f32_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct RopeParams {
            bh: u32,
            td: u32,
            d: u32,
            stride_b: u32,
            src_offset: u32,
            cos_offset: u32,
            sin_offset: u32,
            _pad: u32,
        }

        let params = RopeParams {
            bh: bh as u32,
            td: td as u32,
            d: d as u32,
            stride_b,
            src_offset: l_src.start_offset() as u32,
            cos_offset: l_cos.start_offset() as u32,
            sin_offset: l_sin.start_offset() as u32,
            _pad: 0,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "rope_bf16_params",
        );

        let half_d = d / 2;
        let total_pairs = bh * t * half_d;
        let workgroups = ((total_pairs as u32) + 255) / 256;

        self.device.with_pipeline(ShaderType::RopeBF16, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("rope_bf16_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: cos.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: sin.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: f32_output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rope_bf16_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rope_bf16_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        // Cast F32 → BF16 on GPU
        self.cast_f32_to_bf16_gpu(&f32_output_buffer, el)
    }

    /// BF16 affine on GPU: y = x * mul + add. Input must be contiguous.
    fn affine_bf16_gpu(&self, layout: &Layout, mul_val: f32, add_val: f32) -> Result<Self> {
        let elem_count = layout.shape().elem_count();
        let output_bytes = elem_count * 2;

        let output_buffer = self.device.create_buffer(
            output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "affine_bf16_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct AffineParams {
            elem_count: u32,
            src_offset: u32,
            mul_val: f32,
            add_val: f32,
        }

        let params = AffineParams {
            elem_count: elem_count as u32,
            src_offset: layout.start_offset() as u32,
            mul_val,
            add_val,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "affine_bf16_params",
        );

        let num_pairs = (elem_count + 1) / 2;
        let workgroups = ((num_pairs as u32) + 255) / 256;

        self.device.with_pipeline(ShaderType::AffineBF16, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("affine_bf16_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
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

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("affine_bf16_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("affine_bf16_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            elem_count,
            DType::BF16,
        ))
    }

    /// BF16 unary op on GPU. Input must be contiguous.
    fn unary_bf16_gpu(&self, layout: &Layout, op_type: u32) -> Result<Self> {
        let elem_count = layout.shape().elem_count();
        let output_bytes = elem_count * 2;

        let output_buffer = self.device.create_buffer(
            output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "unary_bf16_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct UnaryParams {
            elem_count: u32,
            op_type: u32,
            src_offset: u32,
            _pad: u32,
        }

        let params = UnaryParams {
            elem_count: elem_count as u32,
            op_type,
            src_offset: layout.start_offset() as u32,
            _pad: 0,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "unary_bf16_params",
        );

        let num_pairs = (elem_count + 1) / 2;
        let workgroups = ((num_pairs as u32) + 255) / 256;

        self.device.with_pipeline(ShaderType::UnaryBF16, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("unary_bf16_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
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

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("unary_bf16_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("unary_bf16_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            elem_count,
            DType::BF16,
        ))
    }

    /// BF16 binary op on GPU. Both inputs must be contiguous.
    fn binary_bf16_gpu(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout, op_type: u32) -> Result<Self> {
        let elem_count = lhs_l.shape().elem_count();
        let output_bytes = elem_count * 2; // BF16 = 2 bytes per element

        let output_buffer = self.device.create_buffer(
            output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "binary_bf16_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct BinaryParams {
            elem_count: u32,
            op_type: u32,
            lhs_offset: u32,
            rhs_offset: u32,
        }

        let params = BinaryParams {
            elem_count: elem_count as u32,
            op_type,
            lhs_offset: lhs_l.start_offset() as u32,
            rhs_offset: rhs_l.start_offset() as u32,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "binary_bf16_params",
        );

        let num_pairs = (elem_count + 1) / 2;
        let workgroups = ((num_pairs as u32) + 255) / 256;

        self.device.with_pipeline(ShaderType::BinaryBF16, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("binary_bf16_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs.buffer.as_entire_binding(),
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

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("binary_bf16_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("binary_bf16_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            elem_count,
            DType::BF16,
        ))
    }

    /// Cast F32 buffer to BF16 on GPU. No CPU readback.
    /// Returns WgpuStorage with DType::BF16 and `elem_count` elements.
    fn cast_f32_to_bf16_gpu(&self, f32_buffer: &wgpu::Buffer, elem_count: usize) -> Result<Self> {
        // BF16 output: ceil(elem_count/2) u32s → elem_count * 2 bytes
        let bf16_bytes = elem_count * 2;
        let output_buffer = self.device.create_buffer(
            bf16_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "cast_f32_to_bf16_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct CastParams {
            elem_count: u32,
        }

        let params = CastParams {
            elem_count: elem_count as u32,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "cast_f32_to_bf16_params",
        );

        // Each thread handles 2 elements (one packed u32)
        let num_pairs = (elem_count + 1) / 2;
        let workgroups = ((num_pairs as u32) + 255) / 256;

        self.device.with_pipeline(ShaderType::CastF32ToBF16, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cast_f32_to_bf16_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: f32_buffer.as_entire_binding(),
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

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cast_f32_to_bf16_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cast_f32_to_bf16_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            elem_count,
            DType::BF16,
        ))
    }

    /// Cast BF16 buffer to F32 on GPU. No CPU readback.
    fn cast_bf16_to_f32_gpu(&self, bf16_buffer: &wgpu::Buffer, elem_count: usize) -> Result<Self> {
        let f32_bytes = elem_count * std::mem::size_of::<f32>();
        let output_buffer = self.device.create_buffer(
            f32_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "cast_bf16_to_f32_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct CastParams {
            elem_count: u32,
        }

        let params = CastParams {
            elem_count: elem_count as u32,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "cast_bf16_to_f32_params",
        );

        let workgroups = ((elem_count as u32) + 255) / 256;

        self.device.with_pipeline(ShaderType::CastBF16ToF32, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cast_bf16_to_f32_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: bf16_buffer.as_entire_binding(),
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

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cast_bf16_to_f32_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("cast_bf16_to_f32_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            elem_count,
            DType::F32,
        ))
    }

    /// F32 reduce (Sum or Max) along last dimension on GPU.
    /// Input: F32 buffer [num_rows, row_size]
    /// Output: F32 buffer [num_rows]
    fn reduce_f32_last_dim_gpu(
        &self,
        f32_buffer: &wgpu::Buffer,
        num_rows: usize,
        row_size: usize,
        shader_type: ShaderType,
    ) -> Result<wgpu::Buffer> {
        let output_bytes = num_rows * std::mem::size_of::<f32>();
        let output_buffer = self.device.create_buffer(
            output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "reduce_f32_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct ReduceParams {
            num_rows: u32,
            row_size: u32,
        }

        let params = ReduceParams {
            num_rows: num_rows as u32,
            row_size: row_size as u32,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "reduce_f32_params",
        );

        self.device.with_pipeline(shader_type, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("reduce_f32_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: f32_buffer.as_entire_binding(),
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

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reduce_f32_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("reduce_f32_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                // One workgroup per row
                pass.dispatch_workgroups(num_rows as u32, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(output_buffer)
    }

    /// BF16 index select (dim=0) on GPU.
    /// Copies rows from BF16 source based on U32 index array.
    fn index_select_bf16_dim0_gpu(
        &self,
        ids: &Self,
        src_layout: &Layout,
        ids_layout: &Layout,
    ) -> Result<Self> {
        let src_dims = src_layout.dims();
        let row_size = if src_dims.len() >= 2 {
            src_dims[1..].iter().product::<usize>()
        } else {
            1
        };
        let num_indices = ids_layout.shape().elem_count();
        let output_elem_count = num_indices * row_size;
        let output_bytes = output_elem_count * 2; // BF16

        let output_buffer = self.device.create_buffer(
            output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "index_select_bf16_output",
        );

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct IndexSelectParams {
            num_indices: u32,
            row_size: u32,
            src_offset: u32,
            ids_offset: u32,
        }

        let params = IndexSelectParams {
            num_indices: num_indices as u32,
            row_size: row_size as u32,
            src_offset: src_layout.start_offset() as u32,
            ids_offset: ids_layout.start_offset() as u32,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "index_select_bf16_params",
        );

        let pairs_per_row = row_size / 2;
        let total_u32s = num_indices * pairs_per_row;
        let workgroups = ((total_u32s as u32) + 255) / 256;

        self.device.with_pipeline(ShaderType::IndexSelectBF16, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("index_select_bf16_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ids.buffer.as_entire_binding(),
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

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("index_select_bf16_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("index_select_bf16_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            output_elem_count,
            DType::BF16,
        ))
    }

    /// Make a non-contiguous tensor contiguous on GPU using strided copy shader.
    /// Returns a new WgpuStorage with contiguous layout.
    fn make_contiguous_gpu(&self, layout: &Layout) -> Result<Self> {
        if layout.is_contiguous() {
            return Ok(self.clone());
        }
        let elem_count = layout.shape().elem_count();
        let elem_size = self.dtype.size_in_bytes();
        let output_bytes = elem_count * elem_size;

        let output_buffer = self.device.create_buffer(
            output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "make_contiguous_output",
        );

        let mut dst = WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            elem_count,
            self.dtype,
        );

        self.copy_strided_src_internal(&mut dst, 0, layout)?;

        Ok(dst)
    }

    /// Internal version of copy_strided_src for use within WgpuStorage methods.
    fn copy_strided_src_internal(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        // Delegate to the BackendStorage implementation
        // We re-implement the logic here to avoid borrow issues
        if src_l.is_contiguous() {
            let elem_size = self.dtype.size_in_bytes();
            let src_byte_offset = (src_l.start_offset() * elem_size) as u64;
            let dst_byte_offset = (dst_offset * elem_size) as u64;
            let copy_size = (src_l.shape().elem_count() * elem_size) as u64;

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_contiguous_internal_encoder"),
            });
            encoder.copy_buffer_to_buffer(&self.buffer, src_byte_offset, &dst.buffer, dst_byte_offset, copy_size);
            self.device.queue().submit(std::iter::once(encoder.finish()));
            return Ok(());
        }

        let elem_count = src_l.shape().elem_count();
        if elem_count == 0 {
            return Ok(());
        }

        let dims = src_l.dims();
        let strides = src_l.stride();
        let ndims = dims.len();
        if ndims > 6 {
            return Err(crate::Error::Msg(
                "copy_strided_src: more than 6 dimensions not supported on wgpu".to_string(),
            ));
        }

        let mut shape_arr = [1u32; 6];
        let mut stride_arr = [0u32; 6];
        for i in 0..ndims {
            shape_arr[i] = dims[i] as u32;
            stride_arr[i] = strides[i] as u32;
        }

        let params = CopyStridedParams {
            ndims: ndims as u32,
            src_offset: src_l.start_offset() as u32,
            dst_offset: dst_offset as u32,
            elem_count: elem_count as u32,
            shape: shape_arr,
            strides: stride_arr,
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "copy_strided_internal_params",
        );

        let shader_type = match self.dtype {
            DType::BF16 => ShaderType::CopyStridedBF16,
            DType::F32 => ShaderType::CopyStridedF32,
            _ => {
                return Err(crate::Error::Msg(format!(
                    "copy_strided_src not implemented on wgpu for dtype {:?}",
                    self.dtype
                )));
            }
        };

        let workgroups = match self.dtype {
            DType::BF16 => {
                let num_pairs = (elem_count + 1) / 2;
                ((num_pairs as u32) + 255) / 256
            }
            _ => ((elem_count as u32) + 255) / 256,
        };

        self.device.with_pipeline(shader_type, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("copy_strided_internal_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_strided_internal_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("copy_strided_internal_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(())
    }

    fn matmul_gpu(&self, rhs: &Self, m: usize, n: usize, k: usize) -> Result<Self> {
        let output_size = m * n;
        let output_bytes = output_size * std::mem::size_of::<f32>();

        let output_buffer = self.device.create_buffer(
            output_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "matmul_output",
        );

        let dims = MatmulDimensions {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            _padding: 0,
        };
        let dims_bytes = bytemuck::bytes_of(&dims);
        let dims_buffer = self.device.create_buffer_init(
            dims_bytes,
            wgpu::BufferUsages::UNIFORM,
            "matmul_dims",
        );

        self.device.with_pipeline(ShaderType::MatmulF32, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: rhs.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("matmul_pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&cached.pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups_x = (m as u32 + 15) / 16;
                let workgroups_y = (n as u32 + 15) / 16;
                compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            output_size,
            DType::F32,
        ))
    }
}

impl BackendStorage for WgpuStorage {
    type Device = WgpuDevice;

    fn try_clone(&self, _layout: &Layout) -> Result<Self> {
        Ok(self.clone())
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match self.dtype {
            DType::U8 => Ok(CpuStorage::U8(self.to_cpu()?)),
            DType::U32 => Ok(CpuStorage::U32(self.to_cpu()?)),
            DType::I16 => Ok(CpuStorage::I16(self.to_cpu()?)),
            DType::I32 => Ok(CpuStorage::I32(self.to_cpu()?)),
            DType::I64 => Ok(CpuStorage::I64(self.to_cpu()?)),
            DType::F16 => {
                let data: Vec<u16> = self.to_cpu()?;
                let data: Vec<half::f16> = data.into_iter().map(half::f16::from_bits).collect();
                Ok(CpuStorage::F16(data))
            }
            DType::BF16 => {
                let data: Vec<u16> = self.to_cpu()?;
                let data: Vec<half::bf16> = data.into_iter().map(half::bf16::from_bits).collect();
                Ok(CpuStorage::BF16(data))
            }
            DType::F32 => Ok(CpuStorage::F32(self.to_cpu()?)),
            DType::F64 => Ok(CpuStorage::F64(self.to_cpu()?)),
            dtype => Err(crate::Error::Msg(
                WgpuError::UnsupportedDType(dtype, "to_cpu_storage").to_string(),
            )),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        if self.dtype == DType::BF16 && layout.is_contiguous() && shader_flags().affine_bf16 {
            eprintln!("[WGPU-TRACE] affine BF16 GPU, shape={:?}", layout.shape());
            return self.affine_bf16_gpu(layout, mul as f32, add as f32);
        }
        eprintln!("[WGPU-TRACE] affine CPU fallback, dtype={:?}", self.dtype);
        self.from_cpu_op(layout, |cpu, l| cpu.affine(l, mul, add))
    }

    fn powf(&self, layout: &Layout, exp: f64) -> Result<Self> {
        self.from_cpu_op(layout, |cpu, l| cpu.powf(l, exp))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.from_cpu_op(layout, |cpu, l| cpu.elu(l, alpha))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        eprintln!("[WGPU-TRACE] reduce_op {:?}, dtype={:?}, shape={:?}, dims={:?}", op, self.dtype, layout.shape(), dims);
        // GPU path for BF16: last-dim-only Sum/Max via composition
        // BF16→F32 cast + F32 reduce shader + F32→BF16 cast
        if self.dtype == DType::BF16 && layout.is_contiguous() && dims.len() == 1 && shader_flags().reduce {
            let ndims = layout.dims().len();
            let reduce_dim = dims[0];
            if reduce_dim == ndims - 1 {
                let shape = layout.shape();
                let last_dim = layout.dims()[ndims - 1];
                let num_rows = shape.elem_count() / last_dim;

                let shader_type = match op {
                    ReduceOp::Sum => ShaderType::ReduceSumLastDim,
                    ReduceOp::Max => ShaderType::ReduceMaxLastDim,
                    ReduceOp::Min => ShaderType::ReduceMaxLastDim, // Min handled below
                    _ => {
                        return self.from_cpu_op(layout, |cpu, l| cpu.reduce_op(op, l, dims));
                    }
                };

                // For Min: negate, reduce max, negate back. Or just use CPU for now.
                if matches!(op, ReduceOp::Min) {
                    return self.from_cpu_op(layout, |cpu, l| cpu.reduce_op(op, l, dims));
                }

                // Cast BF16→F32 on GPU
                let f32_storage = self.cast_bf16_to_f32_gpu(&self.buffer, self.count)?;

                // Reduce F32 on GPU
                let reduced_f32_buffer = self.reduce_f32_last_dim_gpu(
                    f32_storage.buffer(),
                    num_rows,
                    last_dim,
                    shader_type,
                )?;

                // Cast F32→BF16 on GPU
                let bf16_result = self.cast_f32_to_bf16_gpu(&reduced_f32_buffer, num_rows)?;

                return Ok(bf16_result);
            }
        }

        // GPU path for F32: last-dim-only Sum/Max
        if self.dtype == DType::F32 && layout.is_contiguous() && dims.len() == 1 && shader_flags().reduce {
            let ndims = layout.dims().len();
            let reduce_dim = dims[0];
            if reduce_dim == ndims - 1 {
                let shape = layout.shape();
                let last_dim = layout.dims()[ndims - 1];
                let num_rows = shape.elem_count() / last_dim;

                let shader_type = match op {
                    ReduceOp::Sum => ShaderType::ReduceSumLastDim,
                    ReduceOp::Max => ShaderType::ReduceMaxLastDim,
                    _ => {
                        return self.from_cpu_op(layout, |cpu, l| cpu.reduce_op(op, l, dims));
                    }
                };

                let reduced_buffer = self.reduce_f32_last_dim_gpu(
                    &self.buffer,
                    num_rows,
                    last_dim,
                    shader_type,
                )?;

                return Ok(WgpuStorage::new(
                    Arc::new(reduced_buffer),
                    self.device.clone(),
                    num_rows,
                    DType::F32,
                ));
            }
        }

        self.from_cpu_op(layout, |cpu, l| cpu.reduce_op(op, l, dims))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        self.from_cpu_binary_op(rhs, lhs_l, rhs_l, |lhs_cpu, rhs_cpu, ll, rl| {
            lhs_cpu.cmp(op, rhs_cpu, ll, rl)
        })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        eprintln!("[WGPU-TRACE] to_dtype {:?}->{:?} shape={:?} contig={}", self.dtype, dtype, layout.shape(), layout.is_contiguous());
        // GPU-native BF16↔F32 casts (contiguous only)
        if layout.is_contiguous() && shader_flags().cast {
            match (self.dtype, dtype) {
                (DType::BF16, DType::F32) => {
                    return self.cast_bf16_to_f32_gpu(&self.buffer, self.count);
                }
                (DType::F32, DType::BF16) => {
                    return self.cast_f32_to_bf16_gpu(&self.buffer, self.count);
                }
                _ => {}
            }
        }
        // Fallback to CPU for other dtype conversions
        let cpu_storage = self.to_cpu_storage()?;
        let result = cpu_storage.to_dtype(layout, dtype)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        eprintln!("[WGPU-TRACE] unary {} dtype={:?} shape={:?} contig={}", B::NAME, self.dtype, layout.shape(), layout.is_contiguous());
        // GPU path for BF16 contiguous tensors
        if self.dtype == DType::BF16 && layout.is_contiguous() && shader_flags().unary_bf16 {
            let op_type: u32 = match B::NAME {
                "exp" => 0,
                "log" => 1,
                "sin" => 2,
                "cos" => 3,
                "tanh" => 4,
                "neg" => 5,
                "recip" => 6,
                "sqr" => 7,
                "sqrt" => 8,
                "gelu" => 9,
                "relu" => 10,
                "silu" => 11,
                "abs" => 12,
                "ceil" => 13,
                "floor" => 14,
                "round" => 15,
                "sign" => 16,
                _ => {
                    return self.from_cpu_op(layout, |cpu, l| cpu.unary_impl::<B>(l));
                }
            };
            return self.unary_bf16_gpu(layout, op_type);
        }

        self.from_cpu_op(layout, |cpu, l| cpu.unary_impl::<B>(l))
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        eprintln!("[WGPU-TRACE] binary {} dtype={:?} lhs={:?} rhs={:?} contig={}/{}", B::NAME, self.dtype, lhs_l.shape(), rhs_l.shape(), lhs_l.is_contiguous(), rhs_l.is_contiguous());
        // GPU path for BF16 contiguous tensors
        if self.dtype == DType::BF16 && lhs_l.is_contiguous() && rhs_l.is_contiguous() && shader_flags().binary_bf16 {
            let op_type: u32 = match B::NAME {
                "add" => 0,
                "sub" => 1,
                "mul" => 2,
                "div" => 3,
                "minimum" => 4,
                "maximum" => 5,
                _ => {
                    return self.from_cpu_binary_op(rhs, lhs_l, rhs_l, |lc, rc, ll, rl| {
                        lc.binary_impl::<B>(rc, ll, rl)
                    });
                }
            };
            return self.binary_bf16_gpu(rhs, lhs_l, rhs_l, op_type);
        }

        self.from_cpu_binary_op(rhs, lhs_l, rhs_l, |lhs_cpu, rhs_cpu, ll, rl| {
            lhs_cpu.binary_impl::<B>(rhs_cpu, ll, rl)
        })
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let cond_cpu = self.to_cpu_storage()?;
        let t_cpu = t.to_cpu_storage()?;
        let f_cpu = f.to_cpu_storage()?;
        let result = cond_cpu.where_cond(layout, &t_cpu, t_l, &f_cpu, f_l)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Self> {
        let inp_cpu = self.to_cpu_storage()?;
        let kernel_cpu = kernel.to_cpu_storage()?;
        let result = inp_cpu.conv1d(l, &kernel_cpu, kernel_l, params)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        let inp_cpu = self.to_cpu_storage()?;
        let kernel_cpu = kernel.to_cpu_storage()?;
        let result = inp_cpu.conv_transpose1d(l, &kernel_cpu, kernel_l, params)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Self> {
        let inp_cpu = self.to_cpu_storage()?;
        let kernel_cpu = kernel.to_cpu_storage()?;
        let result = inp_cpu.conv2d(l, &kernel_cpu, kernel_l, params)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        let inp_cpu = self.to_cpu_storage()?;
        let kernel_cpu = kernel.to_cpu_storage()?;
        let result = inp_cpu.conv_transpose2d(l, &kernel_cpu, kernel_l, params)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn avg_pool2d(&self, l: &Layout, kernel: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.from_cpu_op(l, |cpu, layout| cpu.avg_pool2d(layout, kernel, stride))
    }

    fn max_pool2d(&self, l: &Layout, kernel: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        self.from_cpu_op(l, |cpu, layout| cpu.max_pool2d(layout, kernel, stride))
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        self.from_cpu_op(l, |cpu, layout| cpu.upsample_nearest1d(layout, sz))
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        self.from_cpu_op(l, |cpu, layout| cpu.upsample_nearest2d(layout, h, w))
    }

    fn upsample_bilinear2d(
        &self,
        l: &Layout,
        h: usize,
        w: usize,
        align_corners: bool,
        scale_h: Option<f64>,
        scale_w: Option<f64>,
    ) -> Result<Self> {
        self.from_cpu_op(l, |cpu, layout| {
            cpu.upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)
        })
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let src_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let result = src_cpu.gather(l, &ids_cpu, ids_l, dim)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        cpu.scatter_set(l, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;

        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device, &cpu)?;
        *self = new_storage;
        Ok(())
    }

    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let mut cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        cpu.scatter_add_set(l, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;

        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device, &cpu)?;
        *self = new_storage;
        Ok(())
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        eprintln!("[WGPU-TRACE] index_select dim={} dtype={:?} src_shape={:?} ids_shape={:?}", dim, self.dtype, l.shape(), ids_l.shape());
        // GPU path for BF16 dim=0 (embedding lookup) with even row_size
        if self.dtype == DType::BF16 && dim == 0 && l.is_contiguous() && ids_l.is_contiguous() && shader_flags().index_select {
            let src_dims = l.dims();
            let row_size = if src_dims.len() >= 2 {
                src_dims[1..].iter().product::<usize>()
            } else {
                1
            };
            // Shader requires even row_size for u32 packing alignment
            if row_size % 2 == 0 {
                return self.index_select_bf16_dim0_gpu(ids, l, ids_l);
            }
        }

        let src_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let result = src_cpu.index_select(&ids_cpu, l, ids_l, dim)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let self_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        let result = self_cpu.index_add(l, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let (b, m, n, k) = bmnk;
        eprintln!("[WGPU-TRACE] matmul dtype={:?} b={} m={} n={} k={} contig={}/{}", self.dtype, b, m, n, k, lhs_l.is_contiguous(), rhs_l.is_contiguous());

        let is_contiguous = lhs_l.is_contiguous() && rhs_l.is_contiguous();
        let is_unbatched = b == 1;

        // BF16 GPU matmul (supports batched)
        if self.dtype == DType::BF16 && is_contiguous && shader_flags().matmul_bf16 {
            return self.matmul_bf16_gpu(rhs, b, m, n, k);
        }

        // F32 GPU matmul
        if self.dtype == DType::F32 && is_contiguous && is_unbatched {
            return self.matmul_gpu(rhs, m, n, k);
        }

        // Non-contiguous BF16: CPU fallback for now (strided copy shader needs debugging)
        // TODO: use make_contiguous_gpu + matmul_bf16_gpu once strided copy BF16 is verified
        if self.dtype == DType::BF16 && !is_contiguous {
            eprintln!("[WGPU-TRACE] matmul non-contiguous BF16: CPU fallback");
            let lhs_cpu = self.to_cpu_storage()?;
            let rhs_cpu = rhs.to_cpu_storage()?;
            let lhs_f32 = lhs_cpu.to_dtype(lhs_l, DType::F32)?;
            let rhs_f32 = rhs_cpu.to_dtype(rhs_l, DType::F32)?;
            let lhs_f32_layout = Layout::contiguous(lhs_l.shape());
            let rhs_f32_layout = Layout::contiguous(rhs_l.shape());
            let result_f32 = lhs_f32.matmul(&rhs_f32, bmnk, &lhs_f32_layout, &rhs_f32_layout)?;
            let out_shape = crate::Shape::from_dims(&[b * m, n]);
            let result_layout = Layout::contiguous(&out_shape);
            let result_bf16 = result_f32.to_dtype(&result_layout, DType::BF16)?;
            return BackendDevice::storage_from_cpu_storage(&self.device, &result_bf16);
        }
        self.from_cpu_binary_op(rhs, lhs_l, rhs_l, |lhs_cpu, rhs_cpu, ll, rl| {
            lhs_cpu.matmul(rhs_cpu, bmnk, ll, rl)
        })
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        eprintln!("[WGPU-TRACE] copy_strided_src dtype={:?} shape={:?} contig={} offset={}", self.dtype, src_l.shape(), src_l.is_contiguous(), dst_offset);
        // If contiguous, use a simple buffer copy
        if src_l.is_contiguous() {
            let elem_size = self.dtype.size_in_bytes();
            let src_byte_offset = (src_l.start_offset() * elem_size) as u64;
            let dst_byte_offset = (dst_offset * elem_size) as u64;
            let copy_size = (src_l.shape().elem_count() * elem_size) as u64;

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_contiguous_encoder"),
            });
            encoder.copy_buffer_to_buffer(&self.buffer, src_byte_offset, &dst.buffer, dst_byte_offset, copy_size);
            self.device.queue().submit(std::iter::once(encoder.finish()));
            return Ok(());
        }

        // Non-contiguous: GPU shader if flag enabled, otherwise CPU fallback
        if shader_flags().copy_strided {
            eprintln!("[WGPU-TRACE] copy_strided_src GPU shader for non-contiguous");
            return self.copy_strided_src_internal(dst, dst_offset, src_l);
        }
        eprintln!("[WGPU-TRACE] copy_strided_src CPU fallback for non-contiguous");
        let src_cpu = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src_cpu.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;
        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device, &dst_cpu)?;
        *dst = new_storage;
        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        // copy2d: for d1 rows, copy d2 elements with src stride=src_s, dst stride=dst_s
        // If strides match d2 (contiguous rows), use a single buffer copy
        if src_s == d2 && dst_s == d2 {
            let elem_size = self.dtype.size_in_bytes();
            let total_elems = d1 * d2;
            let src_byte_offset = (src_o * elem_size) as u64;
            let dst_byte_offset = (dst_o * elem_size) as u64;
            let copy_size = (total_elems * elem_size) as u64;

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy2d_contiguous_encoder"),
            });
            encoder.copy_buffer_to_buffer(&self.buffer, src_byte_offset, &dst.buffer, dst_byte_offset, copy_size);
            self.device.queue().submit(std::iter::once(encoder.finish()));
            return Ok(());
        }

        // Non-contiguous: use strided copy shader with 2D shape
        let params = CopyStridedParams {
            ndims: 2,
            src_offset: src_o as u32,
            dst_offset: dst_o as u32,
            elem_count: (d1 * d2) as u32,
            shape: [d1 as u32, d2 as u32, 1, 1, 1, 1],
            strides: [src_s as u32, 1, 0, 0, 0, 0],
        };
        let params_buffer = self.device.create_buffer_init(
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM,
            "copy2d_strided_params",
        );

        let elem_count = d1 * d2;
        let shader_type = match self.dtype {
            DType::BF16 => ShaderType::CopyStridedBF16,
            DType::F32 => ShaderType::CopyStridedF32,
            _ => {
                let src_cpu = self.to_cpu_storage()?;
                let mut dst_cpu = dst.to_cpu_storage()?;
                src_cpu.copy2d(&mut dst_cpu, d1, d2, src_s, dst_s, src_o, dst_o)?;
                let new_storage = BackendDevice::storage_from_cpu_storage(&self.device, &dst_cpu)?;
                *dst = new_storage;
                return Ok(());
            }
        };

        let workgroups = match self.dtype {
            DType::BF16 => {
                let num_pairs = (elem_count + 1) / 2;
                ((num_pairs as u32) + 255) / 256
            }
            _ => ((elem_count as u32) + 255) / 256,
        };

        self.device.with_pipeline(shader_type, |cached| {
            let bind_group = self.device.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("copy2d_strided_bind_group"),
                layout: &cached.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

            let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy2d_strided_encoder"),
            });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("copy2d_strided_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.device.queue().submit(std::iter::once(encoder.finish()));
        });

        Ok(())
    }

    fn const_set(&mut self, v: crate::scalar::Scalar, l: &Layout) -> Result<()> {
        let mut cpu = self.to_cpu_storage()?;
        cpu.const_set(v, l)?;

        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device, &cpu)?;
        *self = new_storage;
        Ok(())
    }
}
