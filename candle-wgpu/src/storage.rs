//! wgpu storage implementation

use crate::device::WgpuDevice;
use crate::error::WgpuError;
use candle_core::backend::{BackendDevice, BackendStorage};
use candle_core::conv::{ParamsConv1D, ParamsConv2D, ParamsConvTranspose1D, ParamsConvTranspose2D};
use candle_core::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use candle_core::{CpuStorage, DType, Layout, Result};
use std::sync::Arc;
use wgpu::Buffer;

/// Storage for tensors on a wgpu device
#[derive(Debug, Clone)]
pub struct WgpuStorage {
    buffer: Arc<Buffer>,
    device: WgpuDevice,
    count: usize,
    dtype: DType,
}

impl WgpuStorage {
    /// Create new wgpu storage
    pub fn new(buffer: Arc<Buffer>, device: WgpuDevice, count: usize, dtype: DType) -> Self {
        Self {
            buffer,
            device,
            count,
            dtype,
        }
    }

    /// Get the underlying wgpu buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get the element count
    pub fn count(&self) -> usize {
        self.count
    }

    /// Copy data from GPU to CPU
    fn to_cpu<T: bytemuck::Pod>(&self) -> Result<Vec<T>> {
        let size = self.count * std::mem::size_of::<T>();

        // Create staging buffer for readback
        let staging_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_readback"),
            size: size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from storage buffer to staging buffer
        let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size as u64);
        self.device.queue().submit(std::iter::once(encoder.finish()));

        // Map and read the staging buffer
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.device().poll(wgpu::Maintain::Wait);

        rx.recv()
            .map_err(|_| candle_core::Error::Msg(WgpuError::BufferMapFailed.to_string()))?
            .map_err(|_| candle_core::Error::Msg(WgpuError::BufferMapFailed.to_string()))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Create storage from CPU operation result (fallback for unimplemented ops)
    fn from_cpu_op<F>(&self, layout: &Layout, f: F) -> Result<Self>
    where
        F: FnOnce(&CpuStorage, &Layout) -> Result<CpuStorage>,
    {
        let cpu_storage = self.to_cpu_storage()?;
        let result = f(&cpu_storage, layout)?;
        BackendDevice::storage_from_cpu_storage(&self.device, &result)
    }

    /// Create storage from binary CPU operation result (fallback for unimplemented ops)
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
            dtype => Err(candle_core::Error::Msg(
                WgpuError::UnsupportedDType(dtype, "to_cpu_storage").to_string(),
            )),
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        // TODO: Implement GPU shader
        // For now, fallback to CPU
                self.from_cpu_op(layout, |cpu, l| cpu.affine(l, mul, add))
    }

    fn powf(&self, layout: &Layout, exp: f64) -> Result<Self> {
        // TODO: Implement GPU shader
                self.from_cpu_op(layout, |cpu, l| cpu.powf(l, exp))
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        // TODO: Implement GPU shader
                self.from_cpu_op(layout, |cpu, l| cpu.elu(l, alpha))
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, dims: &[usize]) -> Result<Self> {
        // TODO: Implement GPU shader for reduce operations
                self.from_cpu_op(layout, |cpu, l| cpu.reduce_op(op, l, dims))
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        // TODO: Implement GPU shader
                self.from_cpu_binary_op(rhs, lhs_l, rhs_l, |lhs_cpu, rhs_cpu, ll, rl| {
            lhs_cpu.cmp(op, rhs_cpu, ll, rl)
        })
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        // TODO: Implement GPU shader
                let cpu_storage = self.to_cpu_storage()?;
        let result = cpu_storage.to_dtype(layout, dtype)?;
        BackendDevice::storage_from_cpu_storage(&self.device,&result)
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        // TODO: Implement GPU shaders for unary operations
                self.from_cpu_op(layout, |cpu, l| cpu.unary_impl::<B>(l))
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        // TODO: Implement GPU shaders for binary operations
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
        // TODO: Implement GPU shader
                let cond_cpu = self.to_cpu_storage()?;
        let t_cpu = t.to_cpu_storage()?;
        let f_cpu = f.to_cpu_storage()?;
        let result = cond_cpu.where_cond(layout, &t_cpu, t_l, &f_cpu, f_l)?;
        BackendDevice::storage_from_cpu_storage(&self.device,&result)
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv1D,
    ) -> Result<Self> {
        // TODO: Implement GPU shader
                let inp_cpu = self.to_cpu_storage()?;
        let kernel_cpu = kernel.to_cpu_storage()?;
        let result = inp_cpu.conv1d(l, &kernel_cpu, kernel_l, params)?;
        BackendDevice::storage_from_cpu_storage(&self.device,&result)
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose1D,
    ) -> Result<Self> {
        // TODO: Implement GPU shader
                let inp_cpu = self.to_cpu_storage()?;
        let kernel_cpu = kernel.to_cpu_storage()?;
        let result = inp_cpu.conv_transpose1d(l, &kernel_cpu, kernel_l, params)?;
        BackendDevice::storage_from_cpu_storage(&self.device,&result)
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConv2D,
    ) -> Result<Self> {
        // TODO: Implement GPU shader
                let inp_cpu = self.to_cpu_storage()?;
        let kernel_cpu = kernel.to_cpu_storage()?;
        let result = inp_cpu.conv2d(l, &kernel_cpu, kernel_l, params)?;
        BackendDevice::storage_from_cpu_storage(&self.device,&result)
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &ParamsConvTranspose2D,
    ) -> Result<Self> {
        // TODO: Implement GPU shader
                let inp_cpu = self.to_cpu_storage()?;
        let kernel_cpu = kernel.to_cpu_storage()?;
        let result = inp_cpu.conv_transpose2d(l, &kernel_cpu, kernel_l, params)?;
        BackendDevice::storage_from_cpu_storage(&self.device,&result)
    }

    fn avg_pool2d(&self, l: &Layout, kernel: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        // TODO: Implement GPU shader
                self.from_cpu_op(l, |cpu, layout| cpu.avg_pool2d(layout, kernel, stride))
    }

    fn max_pool2d(&self, l: &Layout, kernel: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        // TODO: Implement GPU shader
                self.from_cpu_op(l, |cpu, layout| cpu.max_pool2d(layout, kernel, stride))
    }

    fn upsample_nearest1d(&self, l: &Layout, sz: usize) -> Result<Self> {
        // TODO: Implement GPU shader
                self.from_cpu_op(l, |cpu, layout| cpu.upsample_nearest1d(layout, sz))
    }

    fn upsample_nearest2d(&self, l: &Layout, h: usize, w: usize) -> Result<Self> {
        // TODO: Implement GPU shader
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
        // TODO: Implement GPU shader
                self.from_cpu_op(l, |cpu, layout| {
            cpu.upsample_bilinear2d(layout, h, w, align_corners, scale_h, scale_w)
        })
    }

    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        // TODO: Implement GPU shader
                let src_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let result = src_cpu.gather(l, &ids_cpu, ids_l, dim)?;
        BackendDevice::storage_from_cpu_storage(&self.device,&result)
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
        // TODO: Implement GPU shader
        // For now, do it on CPU and copy back
                let mut cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        cpu.scatter_set(l, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;

        // Copy back to GPU
        
        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device,&cpu)?;
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
        // TODO: Implement GPU shader
                let mut cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        cpu.scatter_add_set(l, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;

        
        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device,&cpu)?;
        *self = new_storage;
        Ok(())
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        // TODO: Implement GPU shader
                let src_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let result = src_cpu.index_select(&ids_cpu, l, ids_l, dim)?;
        BackendDevice::storage_from_cpu_storage(&self.device,&result)
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
        // TODO: Implement GPU shader
                let self_cpu = self.to_cpu_storage()?;
        let ids_cpu = ids.to_cpu_storage()?;
        let src_cpu = src.to_cpu_storage()?;
        let result = self_cpu.index_add(l, &ids_cpu, ids_l, &src_cpu, src_l, dim)?;
        BackendDevice::storage_from_cpu_storage(&self.device,&result)
    }

    fn matmul(
        &self,
        rhs: &Self,
        bmnk: (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        // TODO: Implement GPU shader for matmul - this is the most critical operation!
                self.from_cpu_binary_op(rhs, lhs_l, rhs_l, |lhs_cpu, rhs_cpu, ll, rl| {
            lhs_cpu.matmul(rhs_cpu, bmnk, ll, rl)
        })
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        // TODO: Implement GPU shader
                let src_cpu = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src_cpu.copy_strided_src(&mut dst_cpu, dst_offset, src_l)?;

        
        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device,&dst_cpu)?;
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
        // TODO: Implement GPU shader
                let src_cpu = self.to_cpu_storage()?;
        let mut dst_cpu = dst.to_cpu_storage()?;
        src_cpu.copy2d(&mut dst_cpu, d1, d2, src_s, dst_s, src_o, dst_o)?;

        
        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device,&dst_cpu)?;
        *dst = new_storage;
        Ok(())
    }

    fn const_set(&mut self, v: candle_core::scalar::Scalar, l: &Layout) -> Result<()> {
        // TODO: Implement GPU shader
                let mut cpu = self.to_cpu_storage()?;
        cpu.const_set(v, l)?;

        
        let new_storage = BackendDevice::storage_from_cpu_storage(&self.device,&cpu)?;
        *self = new_storage;
        Ok(())
    }
}
