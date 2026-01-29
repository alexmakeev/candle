//! wgpu device implementation

use super::error::WgpuError;
use super::ops;
use super::quantized;
use super::storage::WgpuStorage;
use crate::backend::BackendDevice;
use crate::{CpuStorage, DType, DeviceLocation, Result, Shape};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue};

/// GPU buffer pool that reuses buffers to avoid per-operation Vulkan allocations.
///
/// Organizes free buffers by size bucket (next power of 2). Only pools STORAGE-class
/// buffers with STORAGE | COPY_SRC | COPY_DST usage flags. Uniform and staging
/// buffers are not pooled (small, different usage patterns).
pub struct BufferPool {
    /// Map from size_bucket -> Vec of free buffers ready for reuse
    free_buffers: HashMap<u64, Vec<wgpu::Buffer>>,
    /// Total number of pool hits (buffer reused)
    hits: u64,
    /// Total number of pool misses (new buffer allocated)
    misses: u64,
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let total_free: usize = self.free_buffers.values().map(|v| v.len()).sum();
        f.debug_struct("BufferPool")
            .field("buckets", &self.free_buffers.len())
            .field("total_free", &total_free)
            .field("hits", &self.hits)
            .field("misses", &self.misses)
            .finish()
    }
}

/// The standard usage flags for pooled storage buffers.
const POOLED_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

impl BufferPool {
    fn new() -> Self {
        Self {
            free_buffers: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Acquire a buffer of at least `size` bytes with STORAGE | COPY_SRC | COPY_DST usage.
    /// Returns a buffer from the pool if available, otherwise allocates a new one.
    /// The returned buffer may be larger than requested (rounded up to size bucket).
    fn acquire(&mut self, size: u64, device: &wgpu::Device) -> wgpu::Buffer {
        let bucket = Self::size_bucket(size);

        if let Some(buffers) = self.free_buffers.get_mut(&bucket) {
            if let Some(buf) = buffers.pop() {
                self.hits += 1;
                if (self.hits + self.misses) % 1000 == 0 {
                    self.print_stats();
                }
                return buf;
            }
        }

        self.misses += 1;
        if (self.hits + self.misses) % 1000 == 0 {
            self.print_stats();
        }

        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled_storage"),
            size: bucket,
            usage: POOLED_USAGE,
            mapped_at_creation: false,
        })
    }

    /// Return a buffer to the pool for future reuse.
    /// The buffer must have been acquired from this pool (STORAGE usage flags).
    fn release(&mut self, buffer: wgpu::Buffer) {
        let bucket = Self::size_bucket(buffer.size());
        self.free_buffers.entry(bucket).or_default().push(buffer);
    }

    /// Round up to next power of 2, minimum 256 bytes.
    fn size_bucket(size: u64) -> u64 {
        let min_size = 256u64;
        let size = size.max(min_size);
        size.next_power_of_two()
    }

    fn print_stats(&self) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            (self.hits as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        let total_free: usize = self.free_buffers.values().map(|v| v.len()).sum();
        let total_free_bytes: u64 = self.free_buffers
            .iter()
            .map(|(bucket, bufs)| bucket * bufs.len() as u64)
            .sum();
        eprintln!(
            "[BUFPOOL] hits={} misses={} rate={:.1}% free_bufs={} free_bytes={:.1}MB buckets={}",
            self.hits,
            self.misses,
            hit_rate,
            total_free,
            total_free_bytes as f64 / 1048576.0,
            self.free_buffers.len(),
        );
    }
}

/// A GPU buffer that returns itself to the BufferPool when dropped.
/// Only buffers that were allocated from the pool (bucket-rounded sizes)
/// are returned; other buffers are dropped normally.
pub struct PooledBuffer {
    buffer: Option<wgpu::Buffer>,
    pool: Arc<Mutex<BufferPool>>,
    /// Whether this buffer was allocated from/through the pool (bucket-rounded size).
    /// Only pooled buffers are returned on drop.
    from_pool: bool,
}

impl PooledBuffer {
    /// Create a pooled buffer wrapper. If `from_pool` is true, the buffer will be
    /// returned to the pool on drop.
    pub fn new(buffer: wgpu::Buffer, pool: Arc<Mutex<BufferPool>>, from_pool: bool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
            from_pool,
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &wgpu::Buffer {
        self.buffer.as_ref().expect("PooledBuffer already dropped")
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("buffer", &self.buffer)
            .finish()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            if self.from_pool {
                self.pool.lock().release(buffer);
            }
            // Otherwise buffer is dropped and deallocated by wgpu
        }
    }
}

/// Cached compute pipeline with its bind group layout
#[derive(Debug)]
pub struct CachedPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

/// Shader types for caching
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum ShaderType {
    MatmulF32,
    MatmulF16,
    MatmulBF16,
    SoftmaxF32,
    SoftmaxFused,
    LayerNormF32,
    RopeF32,
    MatmulQ8_0,
    MatmulQ8_0Dp4a,
    ReduceMaxLastDim,
    ReduceSumLastDim,
    ExpF32,
    BroadcastSub,
    BroadcastDiv,
    SoftmaxBF16,
    LayerNormBF16,
    CastF32ToBF16,
    CastBF16ToF32,
    CopyStridedBF16,
    CopyStridedF32,
    BinaryBF16,
    UnaryBF16,
    AffineBF16,
    RmsNormBF16,
    RopeBF16,
    IndexSelectBF16,
    GemvBF16,
}

/// Unique identifier for a wgpu device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId(usize);

impl DeviceId {
    pub fn ordinal(&self) -> usize {
        self.0
    }
}

/// A wgpu device for GPU compute operations
#[derive(Debug)]
pub struct WgpuDevice {
    id: DeviceId,
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter: Arc<Adapter>,
    seed: Mutex<u64>,
    pipeline_cache: Arc<Mutex<HashMap<ShaderType, CachedPipeline>>>,
    buffer_pool: Arc<Mutex<BufferPool>>,
}

impl Clone for WgpuDevice {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            device: Arc::clone(&self.device),
            queue: Arc::clone(&self.queue),
            adapter: Arc::clone(&self.adapter),
            seed: Mutex::new(*self.seed.lock()),
            pipeline_cache: Arc::clone(&self.pipeline_cache),
            buffer_pool: Arc::clone(&self.buffer_pool),
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
            return Err(crate::Error::Msg(WgpuError::NoAdapter.to_string()));
        }

        let adapter = adapters
            .into_iter()
            .nth(ordinal)
            .ok_or_else(|| {
                crate::Error::Msg(format!("No wgpu adapter found at ordinal {}", ordinal))
            })?;

        let adapter = Arc::new(adapter);

        // Query adapter limits and request maximum supported buffer size
        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_buffer_size = adapter_limits.max_buffer_size;
        required_limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;

        eprintln!(
            "[WGPU] Adapter limits: max_buffer_size={} ({:.0}MB), max_storage_buffer_binding_size={} ({:.0}MB)",
            adapter_limits.max_buffer_size,
            adapter_limits.max_buffer_size as f64 / 1048576.0,
            adapter_limits.max_storage_buffer_binding_size,
            adapter_limits.max_storage_buffer_binding_size as f64 / 1048576.0,
        );

        let (device, queue) = pollster::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("candle-wgpu"),
                        required_features: wgpu::Features::empty(),
                        required_limits,
                        memory_hints: wgpu::MemoryHints::Performance,
                    },
                    None,
                )
                .await
        })
        .map_err(|e| crate::Error::Msg(WgpuError::DeviceRequest(e).to_string()))?;

        eprintln!(
            "[WGPU] Device limits: max_buffer_size={} ({:.0}MB), max_storage_buffer_binding_size={} ({:.0}MB)",
            device.limits().max_buffer_size,
            device.limits().max_buffer_size as f64 / 1048576.0,
            device.limits().max_storage_buffer_binding_size,
            device.limits().max_storage_buffer_binding_size as f64 / 1048576.0,
        );

        Ok(Self {
            id: DeviceId(ordinal),
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter,
            seed: Mutex::new(299792458),
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
            buffer_pool: Arc::new(Mutex::new(BufferPool::new())),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn adapter_info(&self) -> wgpu::AdapterInfo {
        self.adapter.get_info()
    }

    pub fn id(&self) -> DeviceId {
        self.id
    }

    pub fn create_buffer(&self, size: u64, usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
        // Pool only STORAGE-class buffers (the big compute buffers).
        // Check if requested usage is a subset of pooled usage flags.
        if POOLED_USAGE.contains(usage) {
            return self.buffer_pool.lock().acquire(size, &self.device);
        }

        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Wrap a buffer allocated from the pool (bucket-rounded size, returns to pool on drop).
    pub fn wrap_pooled(&self, buffer: wgpu::Buffer) -> Arc<PooledBuffer> {
        Arc::new(PooledBuffer::new(buffer, Arc::clone(&self.buffer_pool), true))
    }

    /// Wrap a non-pooled buffer (exact size, NOT returned to pool on drop).
    pub fn wrap_buffer(&self, buffer: wgpu::Buffer) -> Arc<PooledBuffer> {
        Arc::new(PooledBuffer::new(buffer, Arc::clone(&self.buffer_pool), false))
    }

    pub fn create_buffer_init(&self, data: &[u8], usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: data,
            usage,
        })
    }

    pub fn with_pipeline<F, R>(&self, shader_type: ShaderType, f: F) -> R
    where
        F: FnOnce(&CachedPipeline) -> R,
    {
        let mut cache = self.pipeline_cache.lock();

        if !cache.contains_key(&shader_type) {
            let pipeline = self.compile_pipeline(shader_type);
            cache.insert(shader_type, pipeline);
        }

        f(cache.get(&shader_type).unwrap())
    }

    fn compile_pipeline(&self, shader_type: ShaderType) -> CachedPipeline {
        let (shader_source, label) = match shader_type {
            ShaderType::MatmulF32 => (ops::MATMUL_SHADER, "matmul_f32"),
            ShaderType::MatmulF16 => (ops::MATMUL_SHADER, "matmul_f16"),
            ShaderType::MatmulBF16 => (ops::MATMUL_BF16_SHADER, "matmul_bf16"),
            ShaderType::SoftmaxF32 => (ops::SOFTMAX_SHADER, "softmax_f32"),
            ShaderType::SoftmaxFused => (ops::SOFTMAX_FUSED_SHADER, "softmax_fused"),
            ShaderType::LayerNormF32 => (ops::LAYER_NORM_SHADER, "layer_norm_f32"),
            ShaderType::RopeF32 => (ops::ROPE_SHADER, "rope_f32"),
            ShaderType::MatmulQ8_0 => (quantized::Q8_0_MATMUL_SHADER, "matmul_q8_0"),
            ShaderType::MatmulQ8_0Dp4a => (quantized::Q8_0_MATMUL_DP4A_SHADER, "matmul_q8_0_dp4a"),
            ShaderType::ReduceMaxLastDim => (ops::REDUCE_MAX_LAST_DIM_SHADER, "reduce_max_last_dim"),
            ShaderType::ReduceSumLastDim => (ops::REDUCE_SUM_LAST_DIM_SHADER, "reduce_sum_last_dim"),
            ShaderType::ExpF32 => (ops::EXP_SHADER, "exp_f32"),
            ShaderType::BroadcastSub => (ops::BROADCAST_SUB_SHADER, "broadcast_sub"),
            ShaderType::BroadcastDiv => (ops::BROADCAST_DIV_SHADER, "broadcast_div"),
            ShaderType::SoftmaxBF16 => (ops::SOFTMAX_BF16_SHADER, "softmax_bf16"),
            ShaderType::LayerNormBF16 => (ops::LAYER_NORM_BF16_SHADER, "layer_norm_bf16"),
            ShaderType::CastF32ToBF16 => (ops::CAST_F32_TO_BF16_SHADER, "cast_f32_to_bf16"),
            ShaderType::CastBF16ToF32 => (ops::CAST_BF16_TO_F32_SHADER, "cast_bf16_to_f32"),
            ShaderType::CopyStridedBF16 => (ops::COPY_STRIDED_BF16_SHADER, "copy_strided_bf16"),
            ShaderType::CopyStridedF32 => (ops::COPY_STRIDED_F32_SHADER, "copy_strided_f32"),
            ShaderType::BinaryBF16 => (ops::BINARY_BF16_SHADER, "binary_bf16"),
            ShaderType::UnaryBF16 => (ops::UNARY_BF16_SHADER, "unary_bf16"),
            ShaderType::AffineBF16 => (ops::AFFINE_BF16_SHADER, "affine_bf16"),
            ShaderType::RmsNormBF16 => (ops::RMS_NORM_BF16_SHADER, "rms_norm_bf16"),
            ShaderType::RopeBF16 => (ops::ROPE_BF16_SHADER, "rope_bf16"),
            ShaderType::IndexSelectBF16 => (ops::INDEX_SELECT_BF16_SHADER, "index_select_bf16"),
            ShaderType::GemvBF16 => (ops::GEMV_BF16_SHADER, "gemv_bf16"),
        };

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let bind_group_layout = self.create_bind_group_layout_for_shader(shader_type, label);

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_pipeline_layout", label)),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("{}_pipeline", label)),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        CachedPipeline {
            pipeline,
            bind_group_layout,
        }
    }

    fn create_bind_group_layout_for_shader(&self, shader_type: ShaderType, label: &str) -> wgpu::BindGroupLayout {
        match shader_type {
            ShaderType::MatmulF32 | ShaderType::MatmulF16 | ShaderType::MatmulBF16 | ShaderType::GemvBF16 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, false),
                        uniform_entry(3),
                    ],
                })
            }
            ShaderType::MatmulQ8_0 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, false),
                        uniform_entry(3),
                    ],
                })
            }
            ShaderType::MatmulQ8_0Dp4a => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, false),
                        uniform_entry(4),
                    ],
                })
            }
            ShaderType::ReduceMaxLastDim | ShaderType::ReduceSumLastDim => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, false),
                        uniform_entry(2),
                    ],
                })
            }
            ShaderType::SoftmaxFused | ShaderType::SoftmaxBF16 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, false),
                        uniform_entry(2),
                    ],
                })
            }
            ShaderType::ExpF32 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, false),
                        uniform_entry(2),
                    ],
                })
            }
            ShaderType::BroadcastSub | ShaderType::BroadcastDiv => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, false),
                        uniform_entry(3),
                    ],
                })
            }
            ShaderType::RopeF32 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, false), // q is read_write
                        storage_entry(1, true),
                        storage_entry(2, true),
                        uniform_entry(3),
                    ],
                })
            }
            ShaderType::LayerNormBF16 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, false),
                        uniform_entry(4),
                    ],
                })
            }
            ShaderType::LayerNormF32 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, false),
                        uniform_entry(4),
                    ],
                })
            }
            ShaderType::CopyStridedBF16 | ShaderType::CopyStridedF32 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, false),
                        uniform_entry(2),
                    ],
                })
            }
            ShaderType::BinaryBF16 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),  // lhs
                        storage_entry(1, true),  // rhs
                        storage_entry(2, false), // output
                        uniform_entry(3),        // params
                    ],
                })
            }
            ShaderType::RmsNormBF16 => {
                // input(read), alpha(read), output(write), params(uniform)
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, false),
                        uniform_entry(3),
                    ],
                })
            }
            ShaderType::RopeBF16 => {
                // xs(read), cos(read), sin(read), output(write), params(uniform)
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, true),
                        storage_entry(3, false),
                        uniform_entry(4),
                    ],
                })
            }
            ShaderType::IndexSelectBF16 => {
                // src(read), indices(read), output(write), params(uniform)
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, true),
                        storage_entry(2, false),
                        uniform_entry(3),
                    ],
                })
            }
            ShaderType::UnaryBF16 | ShaderType::AffineBF16 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),  // input
                        storage_entry(1, false), // output
                        uniform_entry(2),        // params
                    ],
                })
            }
            ShaderType::CastF32ToBF16 | ShaderType::CastBF16ToF32 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        storage_entry(0, true),
                        storage_entry(1, false),
                        uniform_entry(2),
                    ],
                })
            }
            _ => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[],
                })
            }
        }
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Calculate buffer size in bytes for a given element count and dtype.
/// BF16/F16 are stored as packed u32 (2 elements per u32), so buffer size
/// must be rounded up to 4-byte boundary for odd element counts.
pub fn buffer_size_bytes(elem_count: usize, dtype: DType) -> usize {
    match dtype {
        DType::BF16 | DType::F16 => ((elem_count + 1) / 2) * 4,
        _ => elem_count * dtype.size_in_bytes(),
    }
}

impl BackendDevice for WgpuDevice {
    type Storage = WgpuStorage;

    fn new(ordinal: usize) -> Result<Self> {
        WgpuDevice::new(ordinal)
    }

    fn location(&self) -> DeviceLocation {
        DeviceLocation::Cuda { gpu_id: self.id.0 } // Temporary: use Cuda location
    }

    fn same_device(&self, other: &Self) -> bool {
        self.id == other.id
    }

    fn zeros_impl(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size_in_bytes = buffer_size_bytes(elem_count, dtype);

        let buffer = self.create_buffer(
            size_in_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "zeros",
        );

        let zeros = vec![0u8; size_in_bytes];
        self.queue.write_buffer(&buffer, 0, &zeros);

        Ok(WgpuStorage::new(
            self.wrap_pooled(buffer),
            self.clone(),
            elem_count,
            dtype,
        ))
    }

    unsafe fn alloc_uninit(&self, shape: &Shape, dtype: DType) -> Result<Self::Storage> {
        let elem_count = shape.elem_count();
        let size_in_bytes = buffer_size_bytes(elem_count, dtype);

        let buffer = self.create_buffer(
            size_in_bytes as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "uninit",
        );

        Ok(WgpuStorage::new(
            self.wrap_pooled(buffer),
            self.clone(),
            elem_count,
            dtype,
        ))
    }

    fn storage_from_slice<T: crate::WithDType>(&self, data: &[T]) -> Result<Self::Storage> {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * std::mem::size_of::<T>(),
            )
        };

        // Pad to u32 boundary for BF16/F16 with odd element count
        let padded_size = buffer_size_bytes(data.len(), T::DTYPE);
        let actual_bytes = if padded_size > bytes.len() {
            let mut padded = bytes.to_vec();
            padded.resize(padded_size, 0);
            padded
        } else {
            bytes.to_vec()
        };

        let buffer = self.create_buffer_init(
            &actual_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "from_slice",
        );

        Ok(WgpuStorage::new(
            self.wrap_buffer(buffer),
            self.clone(),
            data.len(),
            T::DTYPE,
        ))
    }

    fn storage_from_cpu_storage(&self, storage: &CpuStorage) -> Result<Self::Storage> {
        let (bytes, dtype, len) = cpu_storage_to_bytes(storage);

        // BF16/F16: pad to u32 boundary if odd element count
        let padded_size = buffer_size_bytes(len, dtype);
        let actual_bytes = if padded_size > bytes.len() {
            let mut padded = bytes.to_vec();
            padded.resize(padded_size, 0);
            padded
        } else {
            bytes.to_vec()
        };

        let buffer = self.create_buffer_init(
            &actual_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            "from_cpu",
        );

        Ok(WgpuStorage::new(
            self.wrap_buffer(buffer),
            self.clone(),
            len,
            dtype,
        ))
    }

    fn storage_from_cpu_storage_owned(&self, storage: CpuStorage) -> Result<Self::Storage> {
        self.storage_from_cpu_storage(&storage)
    }

    fn rand_uniform(&self, shape: &Shape, dtype: DType, lo: f64, hi: f64) -> Result<Self::Storage> {
        use crate::cpu_backend::CpuDevice;
        let cpu_storage = CpuDevice.rand_uniform(shape, dtype, lo, hi)?;
        self.storage_from_cpu_storage(&cpu_storage)
    }

    fn rand_normal(&self, shape: &Shape, dtype: DType, mean: f64, std: f64) -> Result<Self::Storage> {
        use crate::cpu_backend::CpuDevice;
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
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

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
