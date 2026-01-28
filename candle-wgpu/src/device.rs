//! wgpu device implementation

use crate::error::WgpuError;
use crate::ops;
use crate::quantized;
use crate::storage::WgpuStorage;
use candle_core::backend::BackendDevice;
use candle_core::{CpuStorage, DType, DeviceLocation, Result, Shape};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::{Adapter, Device, Instance, Queue};

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
    /// BF16 matmul (stored as u16, computed as f32)
    MatmulBF16,
    SoftmaxF32,
    /// Fused softmax (more efficient than separate ops)
    SoftmaxFused,
    LayerNormF32,
    RopeF32,
    /// Q8_0 quantized matmul (weights Q8, activations F32)
    MatmulQ8_0,
    /// Q8_0 quantized matmul with DP4a (both weights and activations Q8)
    MatmulQ8_0Dp4a,
    /// Reduce max along last dimension
    ReduceMaxLastDim,
    /// Reduce sum along last dimension
    ReduceSumLastDim,
    /// Element-wise exp
    ExpF32,
    /// Broadcast subtraction (x - value broadcasted)
    BroadcastSub,
    /// Broadcast division (x / value broadcasted)
    BroadcastDiv,
    /// BF16 fused softmax (stored as u16, computed as f32)
    SoftmaxBF16,
    /// BF16 layer normalization (stored as u16, computed as f32)
    LayerNormBF16,
}

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
    /// Cache of compiled compute pipelines
    pipeline_cache: Arc<Mutex<HashMap<ShaderType, CachedPipeline>>>,
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
            pipeline_cache: Arc::new(Mutex::new(HashMap::new())),
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

    /// Execute a closure with access to the compute pipeline
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

    /// Compile a compute pipeline for the given shader type
    fn compile_pipeline(&self, shader_type: ShaderType) -> CachedPipeline {
        let (shader_source, label) = match shader_type {
            ShaderType::MatmulF32 => (ops::MATMUL_SHADER, "matmul_f32"),
            ShaderType::MatmulF16 => (ops::MATMUL_SHADER, "matmul_f16"), // TODO: F16 shader
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
        };

        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout based on shader type
        let bind_group_layout = match shader_type {
            ShaderType::MatmulF32 | ShaderType::MatmulF16 | ShaderType::MatmulBF16 => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // A matrix (input)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // B matrix (input)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // C matrix (output)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Dimensions (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::MatmulQ8_0 => {
                // Q8_0 matmul: weights (Q8), input (F32), output (F32), params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Weights (Q8_0 blocks)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Input (F32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output (F32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::MatmulQ8_0Dp4a => {
                // Q8_0 DP4a: weights (Q8), input (Q8), input_scales, output, params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Weights Q8
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Input Q8
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Input scales
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::ReduceMaxLastDim | ShaderType::ReduceSumLastDim => {
                // Reduce ops: input, output, params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Input
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::SoftmaxFused => {
                // Fused softmax: input, output, params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Input
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::ExpF32 => {
                // Exp: input, output, params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Input
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::BroadcastSub | ShaderType::BroadcastDiv => {
                // Broadcast ops: input, value, output, params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Input
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Value (broadcast source)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::RopeF32 => {
                // RoPE: q (in/out), cos_cache, sin_cache, params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Q tensor (read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Cos cache
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Sin cache
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::SoftmaxBF16 => {
                // BF16 softmax: input (BF16 packed), output (F32), params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Input (BF16 packed as u32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output (F32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::LayerNormBF16 => {
                // BF16 layer norm: input (BF16), gamma (BF16), beta (BF16), output (F32), params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Input (BF16 packed as u32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Gamma (BF16 packed as u32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Beta (BF16 packed as u32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output (F32)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            ShaderType::LayerNormF32 => {
                // LayerNorm: input, gamma, beta, output, params
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[
                        // Input
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Gamma
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Beta
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Params (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            }
            _ => {
                // Generic layout for other shaders - can be specialized later
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{}_bind_group_layout", label)),
                    entries: &[],
                })
            }
        };

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
