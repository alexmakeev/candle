//! WGSL compute shader operations
//!
//! This module contains WGSL shaders for GPU compute operations.
//! Each operation is implemented as a compute shader that can be
//! dispatched on the GPU.
//!
//! ## Priority Operations
//!
//! The following operations are prioritized for GPU implementation:
//!
//! 1. **matmul** - Matrix multiplication, ~90% of LLM compute
//! 2. **softmax** - Attention mechanism core
//! 3. **layer_norm** - Normalization layers
//! 4. **rope** - Rotary positional encoding (Qwen uses this)
//! 5. **reduce** - Sum, mean, max operations
//! 6. **quantized_matmul** - Quantized inference (Q4/Q6/Q8)

// Shader source files will be embedded here
// For now, we provide stub implementations

/// WGSL shader for matrix multiplication
pub const MATMUL_SHADER: &str = r#"
// Matrix multiplication shader
// A[M, K] @ B[K, N] = C[M, N]

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
        let a_idx = row * dims.K + k;
        let b_idx = k * dims.N + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    let c_idx = row * dims.N + col;
    c[c_idx] = sum;
}
"#;

/// WGSL shader for softmax operation
pub const SOFTMAX_SHADER: &str = r#"
// Softmax shader
// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    batch_size: u32,
    seq_len: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

// This is a simplified version - production would need reduction
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let offset = batch_idx * params.seq_len;

    // Find max
    var max_val: f32 = input[offset];
    for (var i: u32 = 1u; i < params.seq_len; i = i + 1u) {
        max_val = max(max_val, input[offset + i]);
    }

    // Compute exp and sum
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.seq_len; i = i + 1u) {
        let exp_val = exp(input[offset + i] - max_val);
        output[offset + i] = exp_val;
        sum = sum + exp_val;
    }

    // Normalize
    for (var i: u32 = 0u; i < params.seq_len; i = i + 1u) {
        output[offset + i] = output[offset + i] / sum;
    }
}
"#;

/// WGSL shader for layer normalization
pub const LAYER_NORM_SHADER: &str = r#"
// Layer normalization shader
// y = (x - mean) / sqrt(var + eps) * gamma + beta

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
}

@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let offset = batch_idx * params.hidden_size;

    // Compute mean
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        mean = mean + input[offset + i];
    }
    mean = mean / f32(params.hidden_size);

    // Compute variance
    var variance: f32 = 0.0;
    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        let diff = input[offset + i] - mean;
        variance = variance + diff * diff;
    }
    variance = variance / f32(params.hidden_size);

    // Normalize
    let std_inv = 1.0 / sqrt(variance + params.eps);
    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        let normalized = (input[offset + i] - mean) * std_inv;
        output[offset + i] = normalized * gamma[i] + beta[i];
    }
}
"#;

/// WGSL shader for RoPE (Rotary Positional Encoding)
pub const ROPE_SHADER: &str = r#"
// Rotary Positional Encoding shader
// Used by Qwen and other modern LLMs

@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read> cos_cache: array<f32>;
@group(0) @binding(2) var<storage, read> sin_cache: array<f32>;

struct Params {
    seq_len: u32,
    head_dim: u32,
    num_heads: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pos = global_id.x;
    let head = global_id.y;
    let dim_pair = global_id.z;

    if (pos >= params.seq_len || head >= params.num_heads || dim_pair >= params.head_dim / 2u) {
        return;
    }

    let base_idx = pos * params.num_heads * params.head_dim + head * params.head_dim;
    let idx0 = base_idx + dim_pair * 2u;
    let idx1 = idx0 + 1u;

    let cos_idx = pos * params.head_dim / 2u + dim_pair;
    let cos_val = cos_cache[cos_idx];
    let sin_val = sin_cache[cos_idx];

    let q0 = q[idx0];
    let q1 = q[idx1];

    q[idx0] = q0 * cos_val - q1 * sin_val;
    q[idx1] = q0 * sin_val + q1 * cos_val;
}
"#;

/// WGSL shader for element-wise addition
pub const ADD_SHADER: &str = r#"
// Element-wise addition shader

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Params {
    size: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    c[idx] = a[idx] + b[idx];
}
"#;

/// WGSL shader for element-wise multiplication
pub const MUL_SHADER: &str = r#"
// Element-wise multiplication shader

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Params {
    size: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    c[idx] = a[idx] * b[idx];
}
"#;

/// WGSL shader for reduce max along last dimension
/// Reduces input[batch, reduce_dim] -> output[batch]
pub const REDUCE_MAX_LAST_DIM_SHADER: &str = r#"
// Reduce Max along last dimension
// Input: [num_rows, row_size]
// Output: [num_rows]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows: u32,
    row_size: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

// Shared memory for reduction within workgroup
var<workgroup> shared_max: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row_idx = wid.x;
    let local_idx = lid.x;

    if (row_idx >= params.num_rows) {
        return;
    }

    let row_offset = row_idx * params.row_size;

    // Each thread finds max for its portion
    var local_max: f32 = -3.4028235e+38; // -FLT_MAX
    var i: u32 = local_idx;
    while (i < params.row_size) {
        local_max = max(local_max, input[row_offset + i]);
        i += 256u;
    }

    shared_max[local_idx] = local_max;
    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_max[local_idx] = max(shared_max[local_idx], shared_max[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes result
    if (local_idx == 0u) {
        output[row_idx] = shared_max[0];
    }
}
"#;

/// WGSL shader for reduce sum along last dimension
/// Reduces input[batch, reduce_dim] -> output[batch]
pub const REDUCE_SUM_LAST_DIM_SHADER: &str = r#"
// Reduce Sum along last dimension
// Input: [num_rows, row_size]
// Output: [num_rows]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows: u32,
    row_size: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

// Shared memory for reduction within workgroup
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row_idx = wid.x;
    let local_idx = lid.x;

    if (row_idx >= params.num_rows) {
        return;
    }

    let row_offset = row_idx * params.row_size;

    // Each thread sums its portion
    var local_sum: f32 = 0.0;
    var i: u32 = local_idx;
    while (i < params.row_size) {
        local_sum += input[row_offset + i];
        i += 256u;
    }

    shared_sum[local_idx] = local_sum;
    workgroupBarrier();

    // Parallel reduction in shared memory
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_sum[local_idx] += shared_sum[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // Thread 0 writes result
    if (local_idx == 0u) {
        output[row_idx] = shared_sum[0];
    }
}
"#;

/// WGSL shader for exp (unary operation)
pub const EXP_SHADER: &str = r#"
// Element-wise exp shader

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    size: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    output[idx] = exp(input[idx]);
}
"#;

/// WGSL shader for broadcast subtraction (x - broadcast_value)
/// x[batch, dim] - value[batch, 1] -> output[batch, dim]
pub const BROADCAST_SUB_SHADER: &str = r#"
// Broadcast subtraction: x - value (broadcasted along last dim)
// Input: [num_rows, row_size]
// Value: [num_rows] (one per row, broadcasted)
// Output: [num_rows, row_size]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows: u32,
    row_size: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = params.num_rows * params.row_size;

    if (idx >= total_size) {
        return;
    }

    let row_idx = idx / params.row_size;
    output[idx] = input[idx] - value[row_idx];
}
"#;

/// WGSL shader for broadcast division (x / broadcast_value)
/// x[batch, dim] / value[batch, 1] -> output[batch, dim]
pub const BROADCAST_DIV_SHADER: &str = r#"
// Broadcast division: x / value (broadcasted along last dim)
// Input: [num_rows, row_size]
// Value: [num_rows] (one per row, broadcasted)
// Output: [num_rows, row_size]

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> value: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows: u32,
    row_size: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = params.num_rows * params.row_size;

    if (idx >= total_size) {
        return;
    }

    let row_idx = idx / params.row_size;
    output[idx] = input[idx] / value[row_idx];
}
"#;

/// WGSL shader for BF16 matrix multiplication
/// BF16 stored as u16 (upper 16 bits of f32), converted to f32 for computation
///
/// A[M, K] @ B[K, N] = C[M, N]
/// Input: 2x BF16 packed per u32 (little-endian), row-major
/// Output: F32 array (will be converted to BF16 on CPU for flexibility)
pub const MATMUL_BF16_SHADER: &str = r#"
// Batched BF16 Matrix Multiplication
// A[b, M, K] @ B[b, K, N] -> C[b, M, N]
// Input: BF16 packed as u32 (2 BF16 per u32, little-endian, row-major)
// Output: F32 array
// Batch dimension via global_id.z

struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
    batch_count: u32,
    a_batch_stride: u32,  // elements per batch in A (M*K)
    b_batch_stride: u32,  // elements per batch in B (K*N)
    c_batch_stride: u32,  // elements per batch in C (M*N)
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

// Convert BF16 (16-bit) to F32
// BF16 = upper 16 bits of f32, so just shift left 16
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let batch = global_id.z;

    if (row >= dims.M || col >= dims.N || batch >= dims.batch_count) {
        return;
    }

    let a_offset = batch * dims.a_batch_stride;
    let b_offset = batch * dims.b_batch_stride;
    let c_offset = batch * dims.c_batch_stride;

    var sum: f32 = 0.0;

    // Compute dot product for C[batch, row, col]
    for (var kk: u32 = 0u; kk < dims.K; kk = kk + 1u) {
        // Read A[batch, row, kk]
        let a_idx = a_offset + row * dims.K + kk;
        let a_packed_idx = a_idx / 2u;
        let a_is_high = (a_idx % 2u) == 1u;
        let a_packed = a[a_packed_idx];
        let a_val = select(
            bf16_to_f32(a_packed & 0xFFFFu),
            bf16_to_f32(a_packed >> 16u),
            a_is_high
        );

        // Read B[batch, kk, col]
        let b_idx = b_offset + kk * dims.N + col;
        let b_packed_idx = b_idx / 2u;
        let b_is_high = (b_idx % 2u) == 1u;
        let b_packed = b[b_packed_idx];
        let b_val = select(
            bf16_to_f32(b_packed & 0xFFFFu),
            bf16_to_f32(b_packed >> 16u),
            b_is_high
        );

        sum = sum + a_val * b_val;
    }

    // Write F32 output
    let c_idx = c_offset + row * dims.N + col;
    c[c_idx] = sum;
}
"#;

/// Specialized GEMV (matrix-vector multiply) shader for m=1 BF16 matmul.
/// A[b, 1, K] @ B[b, K, N] -> C[b, 1, N]
/// Uses 256-thread workgroups with shared memory for the input vector.
/// Each workgroup computes one output element via parallel reduction over K.
/// Vectorized BF16 loads (4 values per vec2<u32> load).
pub const GEMV_BF16_SHADER: &str = r#"
// GEMV: A[b, 1, K] @ B[b, K, N] -> C[b, 1, N]
// Specialized for m=1 case. Each workgroup computes ONE output element C[batch, 0, col].
// 256 threads collaborate on the K-dimension reduction.
// Input: BF16 packed as u32 (2 BF16 per u32, little-endian, row-major)
// Output: F32 array

struct Dimensions {
    M: u32,
    N: u32,
    K: u32,
    batch_count: u32,
    a_batch_stride: u32,  // elements per batch in A (M*K = K for m=1)
    b_batch_stride: u32,  // elements per batch in B (K*N)
    c_batch_stride: u32,  // elements per batch in C (M*N = N for m=1)
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

const WG_SIZE: u32 = 256u;

// Shared memory for partial sums — each thread stores its partial dot product
var<workgroup> shared_partials: array<f32, 256>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let tid = local_id.x;
    let col = wg_id.x + wg_id.y * num_wg.x;  // 2D linearization for large N (>65535)
    let batch = wg_id.z;                       // batch in z dimension

    // Guard: col may exceed N when grid is padded for 2D dispatch
    if (col >= dims.N) {
        return;
    }

    let a_offset = batch * dims.a_batch_stride;
    let b_offset = batch * dims.b_batch_stride;
    let c_offset = batch * dims.c_batch_stride;

    // Each thread accumulates partial dot product over its slice of K.
    // Thread tid processes k indices: tid*4, (tid+WG_SIZE)*4, (tid+2*WG_SIZE)*4, ...
    // Each iteration handles 4 consecutive BF16 elements (2 packed u32 reads for A).
    var partial_sum: f32 = 0.0;

    let num_vec4_chunks = dims.K / 4u;  // number of 4-element chunks in K

    // Vectorized loop: each thread strides by WG_SIZE across vec4 chunks
    var ki: u32 = tid;
    loop {
        if (ki >= num_vec4_chunks) {
            break;
        }
        let k_base = ki * 4u;

        // Load 4 consecutive BF16 values from A vector (2 packed u32)
        // A is row-major [1, K], so elements are contiguous: a_offset + k_base
        let a_packed_base = (a_offset + k_base) / 2u;
        let a_packed0 = a[a_packed_base];
        let a_packed1 = a[a_packed_base + 1u];
        let a0 = bf16_to_f32(a_packed0 & 0xFFFFu);
        let a1 = bf16_to_f32(a_packed0 >> 16u);
        let a2 = bf16_to_f32(a_packed1 & 0xFFFFu);
        let a3 = bf16_to_f32(a_packed1 >> 16u);

        // Load B[k_base+i, col] for i=0..3.  B is row-major [K, N].
        // b_idx = b_offset + (k_base+i) * N + col
        let b_row_base = b_offset + k_base * dims.N + col;
        let b_idx0 = b_row_base;
        let b_idx1 = b_row_base + dims.N;
        let b_idx2 = b_row_base + 2u * dims.N;
        let b_idx3 = b_row_base + 3u * dims.N;

        let b_packed0 = b[b_idx0 / 2u];
        let b0 = select(bf16_to_f32(b_packed0 & 0xFFFFu), bf16_to_f32(b_packed0 >> 16u), (b_idx0 % 2u) == 1u);

        let b_packed1 = b[b_idx1 / 2u];
        let b1 = select(bf16_to_f32(b_packed1 & 0xFFFFu), bf16_to_f32(b_packed1 >> 16u), (b_idx1 % 2u) == 1u);

        let b_packed2 = b[b_idx2 / 2u];
        let b2 = select(bf16_to_f32(b_packed2 & 0xFFFFu), bf16_to_f32(b_packed2 >> 16u), (b_idx2 % 2u) == 1u);

        let b_packed3 = b[b_idx3 / 2u];
        let b3 = select(bf16_to_f32(b_packed3 & 0xFFFFu), bf16_to_f32(b_packed3 >> 16u), (b_idx3 % 2u) == 1u);

        partial_sum = partial_sum + a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;

        ki = ki + WG_SIZE;
    }

    // Handle remaining elements (K not divisible by 4)
    // Only threads with tid < remainder do work; others keep partial_sum as-is.
    let remainder_start = num_vec4_chunks * 4u;
    let k_remainder = dims.K - remainder_start;
    if (tid < k_remainder) {
        let kk = remainder_start + tid;
        let a_elem_idx = a_offset + kk;
        let a_packed = a[a_elem_idx / 2u];
        let a_val = select(bf16_to_f32(a_packed & 0xFFFFu), bf16_to_f32(a_packed >> 16u), (a_elem_idx % 2u) == 1u);

        let b_elem_idx = b_offset + kk * dims.N + col;
        let b_packed = b[b_elem_idx / 2u];
        let b_val = select(bf16_to_f32(b_packed & 0xFFFFu), bf16_to_f32(b_packed >> 16u), (b_elem_idx % 2u) == 1u);

        partial_sum = partial_sum + a_val * b_val;
    }

    // Parallel reduction in shared memory.
    // All 256 threads must participate in every barrier (WGSL requirement).
    shared_partials[tid] = partial_sum;
    workgroupBarrier();

    // Tree reduction: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
    if (tid < 128u) { shared_partials[tid] = shared_partials[tid] + shared_partials[tid + 128u]; }
    workgroupBarrier();
    if (tid < 64u) { shared_partials[tid] = shared_partials[tid] + shared_partials[tid + 64u]; }
    workgroupBarrier();
    if (tid < 32u) { shared_partials[tid] = shared_partials[tid] + shared_partials[tid + 32u]; }
    workgroupBarrier();
    if (tid < 16u) { shared_partials[tid] = shared_partials[tid] + shared_partials[tid + 16u]; }
    workgroupBarrier();
    if (tid < 8u) { shared_partials[tid] = shared_partials[tid] + shared_partials[tid + 8u]; }
    workgroupBarrier();
    if (tid < 4u) { shared_partials[tid] = shared_partials[tid] + shared_partials[tid + 4u]; }
    workgroupBarrier();
    if (tid < 2u) { shared_partials[tid] = shared_partials[tid] + shared_partials[tid + 2u]; }
    workgroupBarrier();
    if (tid < 1u) { shared_partials[tid] = shared_partials[tid] + shared_partials[tid + 1u]; }
    workgroupBarrier();

    // Thread 0 writes the final dot product result
    if (tid == 0u) {
        let c_idx = c_offset + col;
        c[c_idx] = shared_partials[0];
    }
}
"#;

/// BF16 binary operation shader (contiguous inputs only)
/// Supports: Add(0), Sub(1), Mul(2), Div(3), Min(4), Max(5)
/// Each thread handles 2 output elements (one packed u32)
pub const BINARY_BF16_SHADER: &str = r#"
// BF16 binary op: output = lhs OP rhs
// All buffers: BF16 packed as u32 (2 BF16 per u32)
// Contiguous inputs only

@group(0) @binding(0) var<storage, read> lhs: array<u32>;
@group(0) @binding(1) var<storage, read> rhs: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

struct Params {
    elem_count: u32,
    op_type: u32,   // 0=Add, 1=Sub, 2=Mul, 3=Div, 4=Min, 5=Max
    lhs_offset: u32,
    rhs_offset: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16(val: f32) -> u32 {
    return bitcast<u32>(val) >> 16u;
}

fn read_lhs_bf16(idx: u32) -> f32 {
    let packed = lhs[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

fn read_rhs_bf16(idx: u32) -> f32 {
    let packed = rhs[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

fn apply_op(a: f32, b: f32) -> f32 {
    switch params.op_type {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return a / b; }
        case 4u: { return min(a, b); }
        case 5u: { return max(a, b); }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.elem_count) {
        return;
    }

    let a0 = read_lhs_bf16(params.lhs_offset + elem_idx);
    let b0 = read_rhs_bf16(params.rhs_offset + elem_idx);
    let r0 = f32_to_bf16(apply_op(a0, b0));

    var r1: u32 = 0u;
    if (elem_idx + 1u < params.elem_count) {
        let a1 = read_lhs_bf16(params.lhs_offset + elem_idx + 1u);
        let b1 = read_rhs_bf16(params.rhs_offset + elem_idx + 1u);
        r1 = f32_to_bf16(apply_op(a1, b1));
    }

    output[elem_idx / 2u] = r0 | (r1 << 16u);
}
"#;

/// BF16 unary operation shader (contiguous inputs only)
/// Supports: Exp(0), Log(1), Sin(2), Cos(3), Tanh(4), Neg(5), Recip(6),
///           Sqr(7), Sqrt(8), Gelu(9), Relu(10), Silu(11), Abs(12),
///           Ceil(13), Floor(14), Round(15), Sign(16)
/// Each thread handles 2 output elements (one packed u32)
pub const UNARY_BF16_SHADER: &str = r#"
// BF16 unary op: output = OP(input)
// All buffers: BF16 packed as u32 (2 BF16 per u32)
// Contiguous inputs only

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    elem_count: u32,
    op_type: u32,
    src_offset: u32,
    _pad: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16(val: f32) -> u32 {
    return bitcast<u32>(val) >> 16u;
}

fn read_bf16(idx: u32) -> f32 {
    let packed = input[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

// Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu_approx(x: f32) -> f32 {
    let k = 0.7978845608; // sqrt(2/pi)
    let inner = k * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + tanh(inner));
}

fn apply_op(x: f32) -> f32 {
    switch params.op_type {
        case 0u: { return exp(x); }
        case 1u: { return log(x); }
        case 2u: { return sin(x); }
        case 3u: { return cos(x); }
        case 4u: { return tanh(x); }
        case 5u: { return -x; }
        case 6u: { return 1.0 / x; }
        case 7u: { return x * x; }
        case 8u: { return sqrt(x); }
        case 9u: { return gelu_approx(x); }
        case 10u: { return max(0.0, x); }
        case 11u: { return x * sigmoid(x); }
        case 12u: { return abs(x); }
        case 13u: { return ceil(x); }
        case 14u: { return floor(x); }
        case 15u: { return select(x, round(x), true); }  // round
        case 16u: { return sign(x); }
        default: { return 0.0; }
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.elem_count) {
        return;
    }

    let r0 = f32_to_bf16(apply_op(read_bf16(params.src_offset + elem_idx)));

    var r1: u32 = 0u;
    if (elem_idx + 1u < params.elem_count) {
        r1 = f32_to_bf16(apply_op(read_bf16(params.src_offset + elem_idx + 1u)));
    }

    output[elem_idx / 2u] = r0 | (r1 << 16u);
}
"#;

/// BF16 affine shader: y = x * mul + add (contiguous)
/// Each thread handles 2 output elements (one packed u32)
pub const AFFINE_BF16_SHADER: &str = r#"
// BF16 affine: output = input * mul + add

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    elem_count: u32,
    src_offset: u32,
    mul_val: f32,
    add_val: f32,
}

@group(0) @binding(2) var<uniform> params: Params;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16(val: f32) -> u32 {
    return bitcast<u32>(val) >> 16u;
}

fn read_bf16(idx: u32) -> f32 {
    let packed = input[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.elem_count) {
        return;
    }

    let v0 = read_bf16(params.src_offset + elem_idx);
    let r0 = f32_to_bf16(v0 * params.mul_val + params.add_val);

    var r1: u32 = 0u;
    if (elem_idx + 1u < params.elem_count) {
        let v1 = read_bf16(params.src_offset + elem_idx + 1u);
        r1 = f32_to_bf16(v1 * params.mul_val + params.add_val);
    }

    output[elem_idx / 2u] = r0 | (r1 << 16u);
}
"#;

/// BF16 fused softmax shader
/// Input: BF16 packed as u32 (2 BF16 per u32, little-endian)
/// Output: F32 array (will be converted to BF16 on CPU)
/// Computes softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
pub const SOFTMAX_BF16_SHADER: &str = r#"
// BF16 Fused softmax along last dimension
// Input: BF16 packed as u32 (2 BF16 per u32)
// Output: F32 array

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows: u32,
    row_size: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

// Convert BF16 (16-bit) to F32
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

// Read BF16 value at logical index
fn read_bf16(idx: u32) -> f32 {
    let packed_idx = idx / 2u;
    let is_high = (idx % 2u) == 1u;
    let packed = input[packed_idx];
    let bf16_bits = select(
        packed & 0xFFFFu,
        packed >> 16u,
        is_high
    );
    return bf16_to_f32(bf16_bits);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row_idx = wid.x;
    let local_idx = lid.x;

    if (row_idx >= params.num_rows) {
        return;
    }

    let row_offset = row_idx * params.row_size;

    // === Phase 1: Find max ===
    var local_max: f32 = -3.4028235e+38;
    var i: u32 = local_idx;
    while (i < params.row_size) {
        let val = read_bf16(row_offset + i);
        local_max = max(local_max, val);
        i += 256u;
    }

    shared_data[local_idx] = local_max;
    workgroupBarrier();

    // Reduce to find global max
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_data[local_idx] = max(shared_data[local_idx], shared_data[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let row_max = shared_data[0];
    workgroupBarrier();

    // === Phase 2: Compute exp(x - max) and sum ===
    var local_sum: f32 = 0.0;
    i = local_idx;
    while (i < params.row_size) {
        let val = read_bf16(row_offset + i);
        let exp_val = exp(val - row_max);
        output[row_offset + i] = exp_val;
        local_sum += exp_val;
        i += 256u;
    }

    shared_data[local_idx] = local_sum;
    workgroupBarrier();

    // Reduce to find sum
    stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_data[local_idx] += shared_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let row_sum = shared_data[0];
    workgroupBarrier();

    // === Phase 3: Normalize ===
    i = local_idx;
    while (i < params.row_size) {
        output[row_offset + i] /= row_sum;
        i += 256u;
    }
}
"#;

/// Fused softmax shader - does everything in one kernel
/// This is more efficient than separate max/sub/exp/sum/div operations
pub const SOFTMAX_FUSED_SHADER: &str = r#"
// Fused softmax along last dimension
// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
//
// This fused version avoids multiple kernel launches and memory transfers.
// Each workgroup processes one row.

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    num_rows: u32,
    row_size: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row_idx = wid.x;
    let local_idx = lid.x;

    if (row_idx >= params.num_rows) {
        return;
    }

    let row_offset = row_idx * params.row_size;

    // === Phase 1: Find max ===
    var local_max: f32 = -3.4028235e+38;
    var i: u32 = local_idx;
    while (i < params.row_size) {
        local_max = max(local_max, input[row_offset + i]);
        i += 256u;
    }

    shared_data[local_idx] = local_max;
    workgroupBarrier();

    // Reduce to find global max
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_data[local_idx] = max(shared_data[local_idx], shared_data[local_idx + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let row_max = shared_data[0];
    workgroupBarrier();

    // === Phase 2: Compute exp(x - max) and sum ===
    var local_sum: f32 = 0.0;
    i = local_idx;
    while (i < params.row_size) {
        let exp_val = exp(input[row_offset + i] - row_max);
        output[row_offset + i] = exp_val;
        local_sum += exp_val;
        i += 256u;
    }

    shared_data[local_idx] = local_sum;
    workgroupBarrier();

    // Reduce to find sum
    stride = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_data[local_idx] += shared_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let row_sum = shared_data[0];
    workgroupBarrier();

    // === Phase 3: Normalize ===
    i = local_idx;
    while (i < params.row_size) {
        output[row_offset + i] /= row_sum;
        i += 256u;
    }
}
"#;


/// F32 → BF16 cast shader
/// Reads F32 array, writes BF16 packed as u32 (2 BF16 per u32, little-endian)
pub const CAST_F32_TO_BF16_SHADER: &str = r#"
// F32 → BF16 cast
// Input: F32 array (elem_count elements)
// Output: u32 array (packed BF16, ceil(elem_count/2) u32s)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

struct Params {
    elem_count: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

// Convert F32 to BF16 (truncate lower 16 bits)
fn f32_to_bf16(val: f32) -> u32 {
    return bitcast<u32>(val) >> 16u;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Each thread handles one pair of elements (packed into one u32)
    let pair_idx = global_id.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.elem_count) {
        return;
    }

    let low = f32_to_bf16(input[elem_idx]);
    var high: u32 = 0u;
    if (elem_idx + 1u < params.elem_count) {
        high = f32_to_bf16(input[elem_idx + 1u]);
    }

    output[pair_idx] = low | (high << 16u);
}
"#;

/// BF16 → F32 cast shader
/// Reads BF16 packed as u32, writes F32 array
pub const CAST_BF16_TO_F32_SHADER: &str = r#"
// BF16 → F32 cast
// Input: u32 array (packed BF16, 2 BF16 per u32)
// Output: F32 array (elem_count elements)

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
    elem_count: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

// Convert BF16 (16-bit) to F32
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.elem_count) {
        return;
    }

    let packed_idx = idx / 2u;
    let is_high = (idx % 2u) == 1u;
    let packed = input[packed_idx];
    let bf16_bits = select(
        packed & 0xFFFFu,
        packed >> 16u,
        is_high
    );
    output[idx] = bf16_to_f32(bf16_bits);
}
"#;

/// BF16 strided copy shader
/// Copies elements from strided BF16 source to contiguous BF16 destination.
/// Each thread handles one output u32 (2 packed BF16 elements).
/// Supports up to 6 dimensions.
pub const COPY_STRIDED_BF16_SHADER: &str = r#"
// BF16 strided copy: src[strides] → dst[contiguous]
// Each thread writes one output u32 (2 packed BF16 elements)
// Avoids runtime-variable array indexing (RADV driver bug workaround)

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Params {
    ndims: u32,
    src_offset: u32,
    dst_offset: u32,
    elem_count: u32,
    shape_0: u32, shape_1: u32, shape_2: u32, shape_3: u32, shape_4: u32, shape_5: u32,
    stride_0: u32, stride_1: u32, stride_2: u32, stride_3: u32, stride_4: u32, stride_5: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

fn bf16_read_src(idx: u32) -> u32 {
    let packed = src[idx / 2u];
    return select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
}

fn flat_to_src(flat: u32) -> u32 {
    var remaining = flat;
    var src_idx: u32 = params.src_offset;

    if (params.ndims > 5u) {
        src_idx += (remaining % params.shape_5) * params.stride_5;
        remaining = remaining / params.shape_5;
    }
    if (params.ndims > 4u) {
        src_idx += (remaining % params.shape_4) * params.stride_4;
        remaining = remaining / params.shape_4;
    }
    if (params.ndims > 3u) {
        src_idx += (remaining % params.shape_3) * params.stride_3;
        remaining = remaining / params.shape_3;
    }
    if (params.ndims > 2u) {
        src_idx += (remaining % params.shape_2) * params.stride_2;
        remaining = remaining / params.shape_2;
    }
    if (params.ndims > 1u) {
        src_idx += (remaining % params.shape_1) * params.stride_1;
        remaining = remaining / params.shape_1;
    }
    if (params.ndims > 0u) {
        src_idx += (remaining % params.shape_0) * params.stride_0;
    }
    return src_idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    // 2D dispatch: linearize global_id for >65535 workgroup support
    let pair_idx = global_id.x + global_id.y * num_wg.x * 256u;
    let out_elem_0 = params.dst_offset + pair_idx * 2u;

    if (out_elem_0 >= params.dst_offset + params.elem_count) {
        return;
    }

    let src_idx_0 = flat_to_src(pair_idx * 2u);
    let low = bf16_read_src(src_idx_0);

    var high: u32 = 0u;
    if (pair_idx * 2u + 1u < params.elem_count) {
        let src_idx_1 = flat_to_src(pair_idx * 2u + 1u);
        high = bf16_read_src(src_idx_1);
    }

    dst[out_elem_0 / 2u] = low | (high << 16u);
}
"#;

/// F32 strided copy shader
/// Copies elements from strided F32 source to contiguous F32 destination.
/// Supports up to 6 dimensions.
pub const COPY_STRIDED_F32_SHADER: &str = r#"
// F32 strided copy: src[strides] → dst[contiguous]
// Avoids runtime-variable array indexing (RADV driver bug workaround)

@group(0) @binding(0) var<storage, read> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;

struct Params {
    ndims: u32,
    src_offset: u32,
    dst_offset: u32,
    elem_count: u32,
    shape_0: u32, shape_1: u32, shape_2: u32, shape_3: u32, shape_4: u32, shape_5: u32,
    stride_0: u32, stride_1: u32, stride_2: u32, stride_3: u32, stride_4: u32, stride_5: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

fn flat_to_src(flat: u32) -> u32 {
    var remaining = flat;
    var src_idx: u32 = params.src_offset;

    // Unrolled: process dims from innermost to outermost
    // Each dim only participates if ndims > dim_index
    if (params.ndims > 5u) {
        src_idx += (remaining % params.shape_5) * params.stride_5;
        remaining = remaining / params.shape_5;
    }
    if (params.ndims > 4u) {
        src_idx += (remaining % params.shape_4) * params.stride_4;
        remaining = remaining / params.shape_4;
    }
    if (params.ndims > 3u) {
        src_idx += (remaining % params.shape_3) * params.stride_3;
        remaining = remaining / params.shape_3;
    }
    if (params.ndims > 2u) {
        src_idx += (remaining % params.shape_2) * params.stride_2;
        remaining = remaining / params.shape_2;
    }
    if (params.ndims > 1u) {
        src_idx += (remaining % params.shape_1) * params.stride_1;
        remaining = remaining / params.shape_1;
    }
    if (params.ndims > 0u) {
        src_idx += (remaining % params.shape_0) * params.stride_0;
    }
    return src_idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {
    // 2D dispatch: linearize global_id for >65535 workgroup support
    let flat_idx = global_id.x + global_id.y * num_wg.x * 256u;

    if (flat_idx >= params.elem_count) {
        return;
    }

    dst[params.dst_offset + flat_idx] = src[flat_to_src(flat_idx)];
}
"#;

/// BF16 RMS Normalization shader
/// Input: BF16 packed as u32, Alpha (weight): BF16 packed as u32
/// Output: BF16 packed as u32
/// Computes: y = x * rsqrt(mean(x^2) + eps) * alpha
/// One workgroup per row (batch element)
pub const RMS_NORM_BF16_SHADER: &str = r#"
// BF16 RMS Normalization
// Input, alpha: BF16 packed as u32 (2 BF16 per u32)
// Output: BF16 packed as u32

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> alpha: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

struct Params {
    num_rows: u32,
    hidden_size: u32,
    eps: f32,
    input_offset: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16(val: f32) -> u32 {
    return bitcast<u32>(val) >> 16u;
}

fn read_input_bf16(idx: u32) -> f32 {
    let packed = input[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

fn read_alpha_bf16(idx: u32) -> f32 {
    let packed = alpha[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let row_idx = wid.x;
    let local_idx = lid.x;

    if (row_idx >= params.num_rows) {
        return;
    }

    let row_offset = params.input_offset + row_idx * params.hidden_size;

    // Phase 1: Compute sum of squares
    var local_sum_sq: f32 = 0.0;
    var i: u32 = local_idx;
    while (i < params.hidden_size) {
        let val = read_input_bf16(row_offset + i);
        local_sum_sq += val * val;
        i += 256u;
    }

    shared_data[local_idx] = local_sum_sq;
    workgroupBarrier();

    // Parallel reduction for sum of squares
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_data[local_idx] += shared_data[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let rms_inv = inverseSqrt(shared_data[0] / f32(params.hidden_size) + params.eps);
    workgroupBarrier();

    // Phase 2: Normalize and scale by alpha, write BF16 output (pair-based)
    let out_row_offset = row_idx * params.hidden_size;
    i = local_idx * 2u;
    while (i < params.hidden_size) {
        let out_idx = out_row_offset + i;

        let val0 = read_input_bf16(row_offset + i);
        let alpha0 = read_alpha_bf16(i);
        let r0 = f32_to_bf16(val0 * rms_inv * alpha0);

        var r1: u32 = 0u;
        if (i + 1u < params.hidden_size) {
            let val1 = read_input_bf16(row_offset + i + 1u);
            let alpha1 = read_alpha_bf16(i + 1u);
            r1 = f32_to_bf16(val1 * rms_inv * alpha1);
        }

        output[out_idx / 2u] = r0 | (r1 << 16u);
        i += 512u; // 256 threads * 2 elements each
    }
}
"#;

/// BF16 RoPE (Rotary Positional Encoding) shader — writes F32 output
/// Input: BF16 packed src [b*h, t*d], BF16 packed cos/sin [t, d/2]
/// Output: F32 array (will be cast to BF16 by caller)
/// Rotation: dst[i1] = src[i1]*cos - src[i2]*sin
///           dst[i2] = src[i1]*sin + src[i2]*cos
/// where i2 = i1 + d/2 (first/second halves of head_dim)
pub const ROPE_BF16_SHADER: &str = r#"
// BF16→F32 RoPE: src[bh, t*d](BF16) → dst[bh, t*d](F32)
// Each thread handles one rotation pair (i_d in 0..d/2)

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> cos_buf: array<u32>;
@group(0) @binding(2) var<storage, read> sin_buf: array<u32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

struct Params {
    bh: u32,           // b * h
    td: u32,           // t * d
    d: u32,            // head_dim
    stride_b: u32,     // 0 = unbatched cos/sin, >0 for batched
    src_offset: u32,
    cos_offset: u32,
    sin_offset: u32,
    _pad: u32,
}

@group(0) @binding(4) var<uniform> params: Params;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn read_src_bf16(idx: u32) -> f32 {
    let packed = src[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

fn read_cos_bf16(idx: u32) -> f32 {
    let packed = cos_buf[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

fn read_sin_bf16(idx: u32) -> f32 {
    let packed = sin_buf[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_idx = global_id.x;
    let half_d = params.d / 2u;
    let t = params.td / params.d;
    let total_pairs = params.bh * t * half_d;

    if (thread_idx >= total_pairs) {
        return;
    }

    // Decompose: thread_idx → (bh_idx, t_idx, i_d)
    let i_d = thread_idx % half_d;
    let remaining = thread_idx / half_d;
    let t_idx = remaining % t;
    let bh_idx = remaining / t;

    // Source indices
    let base = params.src_offset + bh_idx * params.td;
    let i1 = base + t_idx * params.d + i_d;
    let i2 = i1 + half_d;

    // Cos/sin index
    let cs_offset = select(0u, bh_idx * params.td / 2u, params.stride_b > 0u);
    let i_cs = params.cos_offset + cs_offset + t_idx * half_d + i_d;
    let i_ss = params.sin_offset + cs_offset + t_idx * half_d + i_d;

    let x1 = read_src_bf16(i1);
    let x2 = read_src_bf16(i2);
    let cos_val = read_cos_bf16(i_cs);
    let sin_val = read_sin_bf16(i_ss);

    // Write F32 output (no u32 packing race condition)
    let out_base = bh_idx * params.td;
    let o1 = out_base + t_idx * params.d + i_d;
    let o2 = o1 + half_d;

    dst[o1] = x1 * cos_val - x2 * sin_val;
    dst[o2] = x1 * sin_val + x2 * cos_val;
}
"#;

/// BF16 index select (dim=0) shader
/// Copies rows from BF16 source based on U32 index array.
/// Used for embedding lookup: src[vocab, hidden] + ids[seq_len] → out[seq_len, hidden]
/// Assumes row_size is even (always true for hidden_size in LLMs).
/// Each thread copies one u32 (pair of BF16 elements).
pub const INDEX_SELECT_BF16_SHADER: &str = r#"
// BF16 index select (dim=0): copies full rows by index
// src: BF16 packed as u32 [num_src_rows, row_size/2]
// indices: U32 array [num_indices]
// output: BF16 packed as u32 [num_indices, row_size/2]

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

struct Params {
    num_indices: u32,
    row_size: u32,      // in BF16 elements (must be even)
    src_offset: u32,    // in BF16 elements
    ids_offset: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let u32_idx = global_id.x;
    let pairs_per_row = params.row_size / 2u;
    let total_u32s = params.num_indices * pairs_per_row;

    if (u32_idx >= total_u32s) {
        return;
    }

    let out_row = u32_idx / pairs_per_row;
    let col_pair = u32_idx % pairs_per_row;

    let src_row = indices[params.ids_offset + out_row];
    let src_u32_idx = (params.src_offset / 2u) + src_row * pairs_per_row + col_pair;

    output[u32_idx] = src[src_u32_idx];
}
"#;

/// Fused SiLU(gate) * up for BF16 MLP gate
/// Input: two BF16 buffers (gate, up) of same size
/// Output: one BF16 buffer
/// Each thread processes a pair of elements:
///   output[i] = silu(gate[i]) * up[i]
///   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
/// Fuses two dispatches (unary silu + binary mul) into one.
pub const FUSED_SILU_MUL_BF16_SHADER: &str = r#"
// Fused SiLU(gate) * up for BF16 MLP gate projection
// gate, up: BF16 packed as u32 (2 BF16 per u32)
// output: BF16 packed as u32

@group(0) @binding(0) var<storage, read> gate: array<u32>;
@group(0) @binding(1) var<storage, read> up: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;

struct Params {
    elem_count: u32,
    gate_offset: u32,
    up_offset: u32,
    _pad: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

fn f32_to_bf16(val: f32) -> u32 {
    return bitcast<u32>(val) >> 16u;
}

fn read_gate_bf16(idx: u32) -> f32 {
    let packed = gate[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

fn read_up_bf16(idx: u32) -> f32 {
    let packed = up[idx / 2u];
    let bits = select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
    return bf16_to_f32(bits);
}

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let elem_idx = pair_idx * 2u;

    if (elem_idx >= params.elem_count) {
        return;
    }

    // First element of pair
    let g0 = read_gate_bf16(params.gate_offset + elem_idx);
    let u0 = read_up_bf16(params.up_offset + elem_idx);
    let r0 = f32_to_bf16(g0 * sigmoid(g0) * u0);

    // Second element of pair (bounds-checked)
    var r1: u32 = 0u;
    if (elem_idx + 1u < params.elem_count) {
        let g1 = read_gate_bf16(params.gate_offset + elem_idx + 1u);
        let u1 = read_up_bf16(params.up_offset + elem_idx + 1u);
        r1 = f32_to_bf16(g1 * sigmoid(g1) * u1);
    }

    output[elem_idx / 2u] = r0 | (r1 << 16u);
}
"#;

/// BF16 Layer Normalization shader
/// Input: BF16 packed as u32 (2 BF16 per u32, little-endian)
/// Gamma, Beta: BF16 packed as u32
/// Output: F32 array (will be converted to BF16 on CPU)
/// Computes: y = (x - mean) / sqrt(var + eps) * gamma + beta
pub const LAYER_NORM_BF16_SHADER: &str = r#"
// BF16 Layer Normalization
// Input, gamma, beta: BF16 packed as u32 (2 BF16 per u32)
// Output: F32 array

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read> gamma: array<u32>;
@group(0) @binding(2) var<storage, read> beta: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

struct Params {
    batch_size: u32,
    hidden_size: u32,
    eps: f32,
    _padding: u32,
}

@group(0) @binding(4) var<uniform> params: Params;

// Convert BF16 (16-bit) to F32
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

// Read BF16 value at logical index from input
fn read_input_bf16(idx: u32) -> f32 {
    let packed_idx = idx / 2u;
    let is_high = (idx % 2u) == 1u;
    let packed = input[packed_idx];
    let bf16_bits = select(
        packed & 0xFFFFu,
        packed >> 16u,
        is_high
    );
    return bf16_to_f32(bf16_bits);
}

// Read BF16 value from gamma
fn read_gamma_bf16(idx: u32) -> f32 {
    let packed_idx = idx / 2u;
    let is_high = (idx % 2u) == 1u;
    let packed = gamma[packed_idx];
    let bf16_bits = select(
        packed & 0xFFFFu,
        packed >> 16u,
        is_high
    );
    return bf16_to_f32(bf16_bits);
}

// Read BF16 value from beta
fn read_beta_bf16(idx: u32) -> f32 {
    let packed_idx = idx / 2u;
    let is_high = (idx % 2u) == 1u;
    let packed = beta[packed_idx];
    let bf16_bits = select(
        packed & 0xFFFFu,
        packed >> 16u,
        is_high
    );
    return bf16_to_f32(bf16_bits);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let offset = batch_idx * params.hidden_size;

    // Compute mean
    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        mean = mean + read_input_bf16(offset + i);
    }
    mean = mean / f32(params.hidden_size);

    // Compute variance
    var variance: f32 = 0.0;
    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        let diff = read_input_bf16(offset + i) - mean;
        variance = variance + diff * diff;
    }
    variance = variance / f32(params.hidden_size);

    // Normalize and apply scale/shift
    let std_inv = 1.0 / sqrt(variance + params.eps);
    for (var i: u32 = 0u; i < params.hidden_size; i = i + 1u) {
        let normalized = (read_input_bf16(offset + i) - mean) * std_inv;
        let gamma_val = read_gamma_bf16(i);
        let beta_val = read_beta_bf16(i);
        output[offset + i] = normalized * gamma_val + beta_val;
    }
}
"#;
