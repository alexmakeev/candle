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

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

struct Params {
    ndims: u32,
    src_offset: u32,
    dst_offset: u32,
    elem_count: u32,
    // Shape dims (up to 6)
    shape_0: u32, shape_1: u32, shape_2: u32, shape_3: u32, shape_4: u32, shape_5: u32,
    // Source strides (up to 6)
    stride_0: u32, stride_1: u32, stride_2: u32, stride_3: u32, stride_4: u32, stride_5: u32,
}

@group(0) @binding(2) var<uniform> params: Params;

fn bf16_read(buf: ptr<storage, array<u32>, read>, idx: u32) -> u32 {
    let packed = (*buf)[idx / 2u];
    return select(packed & 0xFFFFu, packed >> 16u, (idx % 2u) == 1u);
}

fn flat_to_strided_idx(flat: u32) -> u32 {
    var remaining = flat;
    var src_idx: u32 = params.src_offset;
    let shapes = array<u32, 6>(params.shape_0, params.shape_1, params.shape_2, params.shape_3, params.shape_4, params.shape_5);
    let strides = array<u32, 6>(params.stride_0, params.stride_1, params.stride_2, params.stride_3, params.stride_4, params.stride_5);

    // Decompose flat index into multi-dim and apply strides
    for (var d: i32 = i32(params.ndims) - 1; d >= 0; d = d - 1) {
        let dim_idx = remaining % shapes[d];
        remaining = remaining / shapes[d];
        src_idx += dim_idx * strides[d];
    }
    return src_idx;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let out_elem_0 = params.dst_offset + pair_idx * 2u;

    if (out_elem_0 >= params.dst_offset + params.elem_count) {
        return;
    }

    // First element
    let src_idx_0 = flat_to_strided_idx(pair_idx * 2u);
    let low = bf16_read(&src, src_idx_0);

    // Second element (if exists)
    var high: u32 = 0u;
    if (pair_idx * 2u + 1u < params.elem_count) {
        let src_idx_1 = flat_to_strided_idx(pair_idx * 2u + 1u);
        high = bf16_read(&src, src_idx_1);
    }

    dst[out_elem_0 / 2u] = low | (high << 16u);
}
"#;

/// F32 strided copy shader
/// Copies elements from strided F32 source to contiguous F32 destination.
/// Supports up to 6 dimensions.
pub const COPY_STRIDED_F32_SHADER: &str = r#"
// F32 strided copy: src[strides] → dst[contiguous]

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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let flat_idx = global_id.x;

    if (flat_idx >= params.elem_count) {
        return;
    }

    var remaining = flat_idx;
    var src_idx: u32 = params.src_offset;
    let shapes = array<u32, 6>(params.shape_0, params.shape_1, params.shape_2, params.shape_3, params.shape_4, params.shape_5);
    let strides = array<u32, 6>(params.stride_0, params.stride_1, params.stride_2, params.stride_3, params.stride_4, params.stride_5);

    for (var d: i32 = i32(params.ndims) - 1; d >= 0; d = d - 1) {
        let dim_idx = remaining % shapes[d];
        remaining = remaining / shapes[d];
        src_idx += dim_idx * strides[d];
    }

    dst[params.dst_offset + flat_idx] = src[src_idx];
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
