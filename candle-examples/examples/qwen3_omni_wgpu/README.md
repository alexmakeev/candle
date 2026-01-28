# Qwen3-Omni WGPU Example

Example demonstrating Qwen3-Omni inference using the WGPU backend.

## Overview

This example shows how to use `candle-wgpu` with the Qwen3-Omni model for text generation. Currently, the model loads on CPU (due to VarBuilder limitations), but the WgpuDevice is initialized and available for GPU operations.

## Architecture Note

**Important limitation**: The current Candle architecture doesn't include a `Device::Wgpu` variant. This means:

1. Model weights must be loaded on CPU using `VarBuilder` (which only supports CPU/CUDA/Metal devices)
2. WgpuDevice can be initialized separately for custom GPU operations
3. Full GPU model loading will require extending `candle-core::Device` enum

## Usage

```bash
cargo run --example qwen3_omni_wgpu -- \
  --weight-path /path/to/model \
  --prompt "What is deep learning?"
```

## Requirements

- WGPU-compatible GPU (Vulkan/Metal/DX12)
- Model weights in safetensors format

## Implementation Details

### Current Approach

```rust
// 1. Initialize WgpuDevice (if available)
let wgpu_device = WgpuDevice::new(0)?;
println!("GPU: {:?}", wgpu_device.adapter_info());

// 2. Load model on CPU (VarBuilder limitation)
let device = Device::Cpu;
let vb = VarBuilder::from_mmaped_safetensors(&files, DType::F32, &device)?;
let model = Thinker::new(&config, vb)?;

// 3. Run inference on CPU
// (WgpuDevice available for custom operations)
```

### Future Improvements

To enable full GPU model loading:

1. Extend `candle-core::Device` enum:
```rust
pub enum Device {
    Cpu,
    Cuda(CudaDevice),
    Metal(MetalDevice),
    Wgpu(WgpuDevice),  // Add this variant
}
```

2. Extend `Storage` enum:
```rust
pub enum Storage {
    Cpu(CpuStorage),
    Cuda(CudaStorage),
    Metal(MetalStorage),
    Wgpu(WgpuStorage),  // Add this variant
}
```

3. Update VarBuilder to support WgpuDevice

## Testing

The example will:
- Initialize WgpuDevice and print GPU info
- Load the model on CPU
- Run text generation inference
- Report tokens/second

## See Also

- `/candle-wgpu` - WGPU backend implementation
- `/candle-examples/examples/qwen3_omni_text` - Original CPU example
