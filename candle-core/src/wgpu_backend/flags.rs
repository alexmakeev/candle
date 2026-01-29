//! Per-shader GPU/CPU toggle flags for debugging wgpu shaders individually.
//!
//! Usage:
//!   WGPU_GPU_ALL=0 — disable all GPU shaders (CPU fallback for everything)
//!   WGPU_GPU_ALL=0 WGPU_GPU_MATMUL_BF16=1 — enable only BF16 matmul on GPU
//!   (default: all GPU shaders enabled)
//!
//! Individual flags:
//!   WGPU_GPU_MATMUL_BF16  — BF16 matmul (batched, contiguous)
//!   WGPU_GPU_BINARY_BF16  — binary ops: add, sub, mul, div, min, max
//!   WGPU_GPU_UNARY_BF16   — unary ops: exp, silu, neg, etc.
//!   WGPU_GPU_AFFINE_BF16  — affine: y = x * mul + add
//!   WGPU_GPU_CAST         — BF16↔F32 cast shaders
//!   WGPU_GPU_REDUCE       — reduce Sum/Max (last dim)
//!   WGPU_GPU_INDEX_SELECT — index_select dim=0 (embedding lookup)
//!   WGPU_GPU_COPY_STRIDED — non-contiguous strided copy
//!   WGPU_GPU_RMS_NORM     — RMS normalization
//!   WGPU_GPU_SOFTMAX      — softmax
//!   WGPU_GPU_ROPE         — rotary positional embedding

use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct WgpuShaderFlags {
    pub matmul_bf16: bool,
    pub binary_bf16: bool,
    pub unary_bf16: bool,
    pub affine_bf16: bool,
    pub cast: bool,
    pub reduce: bool,
    pub index_select: bool,
    pub copy_strided: bool,
    pub rms_norm: bool,
    pub softmax: bool,
    pub rope: bool,
}

impl WgpuShaderFlags {
    fn from_env() -> Self {
        // WGPU_GPU_ALL sets the default for all flags.
        // Default is false (all CPU) for safe debugging.
        // Set WGPU_GPU_ALL=1 to enable all GPU shaders at once.
        let default = env_bool("WGPU_GPU_ALL", false);

        let flags = Self {
            matmul_bf16: env_bool("WGPU_GPU_MATMUL_BF16", default),
            binary_bf16: env_bool("WGPU_GPU_BINARY_BF16", default),
            unary_bf16: env_bool("WGPU_GPU_UNARY_BF16", default),
            affine_bf16: env_bool("WGPU_GPU_AFFINE_BF16", default),
            cast: env_bool("WGPU_GPU_CAST", default),
            reduce: env_bool("WGPU_GPU_REDUCE", default),
            index_select: env_bool("WGPU_GPU_INDEX_SELECT", default),
            copy_strided: env_bool("WGPU_GPU_COPY_STRIDED", default),
            rms_norm: env_bool("WGPU_GPU_RMS_NORM", default),
            softmax: env_bool("WGPU_GPU_SOFTMAX", default),
            rope: env_bool("WGPU_GPU_ROPE", default),
        };

        eprintln!("[WGPU-FLAGS] Shader flags: {:?}", flags);
        flags
    }
}

fn env_bool(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(val) => matches!(val.as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => default,
    }
}

static FLAGS: OnceLock<WgpuShaderFlags> = OnceLock::new();

/// Get the global shader flags (initialized once from env vars).
pub fn shader_flags() -> &'static WgpuShaderFlags {
    FLAGS.get_or_init(WgpuShaderFlags::from_env)
}
