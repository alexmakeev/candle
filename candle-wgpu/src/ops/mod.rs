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
