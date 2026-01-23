//! wgpu device implementation

use crate::error::WgpuError;
use crate::storage::WgpuStorage;
use candle_core::backend::BackendDevice;
use candle_core::{CpuStorage, DType, DeviceLocation, Result, Shape};
use parking_lot::Mutex;
use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue};

/// Unique identifier for a wgpu device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

/// A wgpu device for GPU compute operations
#[derive(Debug)]
pub struct WgpuDevice {
    id: DeviceId,
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter: Arc<Adapter>,
    seed: Mutex<u64>,
}

impl Clone for WgpuDevice {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            device: Arc::clone(&self.device),
            queue: Arc::clone(&self.queue),
            adapter: Arc::clone(&self.adapter),
            seed: Mutex::new(*self.seed.lock()),
        }
    }
}

impl WgpuDevice {
    /// Create a new wgpu device
    pub fn new(ordinal: usize) -> Result<Self> {
        let instance = Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());

        if adapters.is_empty() {
            return Err(candle_core::Error::Msg(
                WgpuError::NoAdapter.to_string(),
            ));
        }

        let adapter = adapters
            .into_iter()
            .nth(ordinal)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "No wgpu adapter found at ordinal {}",
                    ordinal
                ))
            })?;

        let adapter = Arc::new(adapter);

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("candle-wgpu"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
        })
        .map_err(|e| candle_core::Error::Msg(WgpuError::DeviceRequest(e).to_string()))?;

        Ok(Self {
            id: DeviceId(ordinal),
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter,
            seed: Mutex::new(299792458), // speed of light as default seed
        })
    }

    /// Get the wgpu device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the wgpu queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Get the wgpu adapter info
    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    /// Get the device id
    pub fn id(&self) -> DeviceId {
        self.id
    }

    /// Create a new buffer with the given size
    pub fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Create a buffer initialized with data
    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        })
    }
}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(ordinal: usize) -> Result<Self> {
        WgpuDevice::new(ordinal)
    }

    fn location(&self) -> DeviceLocation {
        // Use a new DeviceLocation variant or map to existing
        // For now, we'll need to add Wgpu variant to DeviceLocation
        // Temporarily using Cuda as placeholder until we modify candle-core
        DeviceLocation::Cuda { gpu_id: self.id.0 }
    }

    fn same_device(&self, other: &Self) -> bool {
        self.id == other.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size_in_bytes = elem_count * dtype.size_in_bytes();

        // Create a zeroed buffer
        let buffer = self.create_buffer(
            size_in_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "zeros",
        );

        // Zero the buffer by writing zeros
        let zeros = vec![0u8; size_in_bytes];
        self.queue.write_buffer(&buffer, 0, &zeros);

        Ok(WgpuStorage::new(
            Arc::new(buffer),
            self.clone(),
            elem_count,
            dtype,
        ))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size_in_bytes = elem_count * dtype.size_in_bytes();

        let buffer = self.create_buffer(
            size_in_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "uninit",
        );

        Ok(WgpuStorage::new(
            Arc::new(buffer),
            self.clone(),
            elem_count,
            dtype,
        ))
    }

    fn storage_from_slice<T: candle_core::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        let buffer = self.create_buffer_init(
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "from_slice",
        );

        Ok(WgpuStorage::new(
            Arc::new(buffer),
            self.clone(),
            data.len(),
            T::DTYPE,
        ))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let (bytes, dtype, len) = cpu_storage_to_bytes(storage);

        let buffer = self.create_buffer_init(
            bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "from_cpu",
        );

        Ok(WgpuStorage::new(
            Arc::new(buffer),
            self.clone(),
            len,
            dtype,
        ))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        // For now, generate on CPU and transfer
        // TODO: Implement GPU-side random generation
        use candle_core::cpu_backend::CpuDevice;
        
        let cpu_storage = CpuDevice.rand_uniform(shape, dtype, lo, hi)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        // For now, generate on CPU and transfer
        // TODO: Implement GPU-side random generation
        use candle_core::cpu_backend::CpuDevice;
        
        let cpu_storage = CpuDevice.rand_normal(shape, dtype, mean, std)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn set_seed(&self, seed: u64) -> Result<()> {
        *self.seed.lock() = seed;
        Ok(())
    }

    fn get_current_seed(&self) -> Result<u64> {
        Ok(*self.seed.lock())
    }

    fn synchronize(&self) -> Result<()> {
        // wgpu automatically synchronizes when reading back data
        // For explicit sync, we can submit an empty command buffer and wait
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

/// Convert CpuStorage to raw bytes
fn cpu_storage_to_bytes(storage: &CpuStorage) -> (&[u8], DType, usize) {
    match storage {
        CpuStorage::U8(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) },
            DType::U8,
            v.len(),
        ),
        CpuStorage::U32(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) },
            DType::U32,
            v.len(),
        ),
        CpuStorage::I16(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) },
            DType::I16,
            v.len(),
        ),
        CpuStorage::I32(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) },
            DType::I32,
            v.len(),
        ),
        CpuStorage::I64(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) },
            DType::I64,
            v.len(),
        ),
        CpuStorage::F16(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) },
            DType::F16,
            v.len(),
        ),
        CpuStorage::BF16(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) },
            DType::BF16,
            v.len(),
        ),
        CpuStorage::F32(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) },
            DType::F32,
            v.len(),
        ),
        CpuStorage::F64(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) },
            DType::F64,
            v.len(),
        ),
        CpuStorage::F8E4M3(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) },
            DType::F8E4M3,
            v.len(),
        ),
        // Dummy types stored as raw bytes
        CpuStorage::F6E2M3(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) },
            DType::F6E2M3,
            v.len(),
        ),
        CpuStorage::F6E3M2(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) },
            DType::F6E3M2,
            v.len(),
        ),
        CpuStorage::F4(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) },
            DType::F4,
            v.len(),
        ),
        CpuStorage::F8E8M0(v) => (
            unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len()) },
            DType::F8E8M0,
            v.len(),
        ),
    }
}
