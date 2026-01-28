//! wgpu storage implementation

use super::device::{ShaderType, WgpuDevice};
use super::error::WgpuError;
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

        // Read back F32 results and convert to BF16
        let staging_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_bf16_staging"),
            size: total_output_f32_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_bf16_readback"),
        });
        encoder.copy_buffer_to_buffer(&output_f32_buffer, 0, &staging_buffer, 0, total_output_f32_bytes as u64);
        self.device.queue().submit(std::iter::once(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.device().poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| crate::Error::Msg("BF16 matmul: channel recv error".to_string()))?
            .map_err(|_| crate::Error::Msg("BF16 matmul: buffer map error".to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let f32_data: &[f32] = bytemuck::cast_slice(&data);

        // Convert F32 -> BF16 and create output buffer
        let bf16_data: Vec<half::bf16> = f32_data.iter().map(|&x| half::bf16::from_f32(x)).collect();
        drop(data);
        staging_buffer.unmap();

        let bf16_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(bf16_data.as_ptr() as *const u8, bf16_data.len() * 2)
        };
        let output_buffer = self.device.create_buffer_init(
            bf16_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "matmul_bf16_output",
        );

        Ok(WgpuStorage::new(
            Arc::new(output_buffer),
            self.device.clone(),
            total_output_size,
            DType::BF16,
        ))
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
        self.from_cpu_op(layout, |cpu, l| cpu.affine(l, mul, add))
    }

    fn powf(&self, layout: &Layout, exp: f64) -> Result<Self> {
        self.from_cpu_op(layout, |cpu, l| cpu.powf(l, exp))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        self.from_cpu_op(layout, |cpu, l| cpu.elu(l, alpha))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        self.from_cpu_op(layout, |cpu, l| cpu.reduce_op(op, l, dims))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        self.from_cpu_binary_op(rhs, lhs_l, rhs_l, |lhs_cpu, rhs_cpu, ll, rl| {
            lhs_cpu.cmp(op, rhs_cpu, ll, rl)
        })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let cpu_storage = self.to_cpu_storage()?;
        let result = cpu_storage.to_dtype(layout, dtype)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        self.from_cpu_op(layout, |cpu, l| cpu.unary_impl::<B>(l))
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
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

        let is_contiguous = lhs_l.is_contiguous() && rhs_l.is_contiguous();
        let is_unbatched = b == 1;

        // BF16 GPU matmul (supports batched)
        if self.dtype == DType::BF16 && is_contiguous {
            return self.matmul_bf16_gpu(rhs, b, m, n, k);
        }

        // F32 GPU matmul
        if self.dtype == DType::F32 && is_contiguous && is_unbatched {
            return self.matmul_gpu(rhs, m, n, k);
        }

        // Fall back to CPU (BF16 requires F32 conversion since CPU doesn't support BF16 matmul)
        if self.dtype == DType::BF16 {
            let lhs_cpu = self.to_cpu_storage()?;
            let rhs_cpu = rhs.to_cpu_storage()?;
            // to_dtype produces contiguous output regardless of input layout
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
        let src_cpu = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src_cpu.copy2d(&mut dst_cpu, d1, d2, src_s, dst_s, src_o, dst_o)?;

        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device, &dst_cpu)?;
        *dst = new_storage;
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
