# WGPU Example Build and Test Guide

## Overview

This guide explains how to build and test the `qwen3_omni_wgpu` example on the remote machine (Lyuda).

## Quick Start

### On Local Machine (jam)

```bash
cd /home/alexii/lluda/candle-16b
git add .
git commit -m "Add qwen3_omni_wgpu example"
git push origin qwen3-omni-16b
```

### On Remote Machine (Lyuda)

```bash
# Connect via SSH
sshpass -p '1q2w3e' ssh -p 2233 lluda@127.0.0.1

# Update code
cd ~/candle-16b
git pull origin qwen3-omni-16b

# Build example
./test_wgpu_example.sh

# Or manually:
cargo build --example qwen3_omni_wgpu --release
```

## Running the Example

### Basic Usage

```bash
cargo run --example qwen3_omni_wgpu --release -- \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni \
  --prompt "Explain what is deep learning in simple terms"
```

### With Parameters

```bash
cargo run --example qwen3_omni_wgpu --release -- \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni \
  --prompt "What is machine learning?" \
  --sample-len 200 \
  --temperature 0.7 \
  --top-p 0.9 \
  --repeat-penalty 1.1
```

### Force CPU Mode

```bash
cargo run --example qwen3_omni_wgpu --release -- \
  --cpu \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni \
  --prompt "Test prompt"
```

## What the Example Does

1. **Initializes WgpuDevice**
   - Detects GPU (Radeon 8060S)
   - Prints adapter information
   - Falls back to CPU if WGPU unavailable

2. **Loads Model on CPU**
   - Due to VarBuilder limitation
   - Uses memory-mapped safetensors
   - Loads BF16 or F32 weights

3. **Runs Inference**
   - Text generation loop
   - Token-by-token output
   - Performance metrics (tokens/sec)

## Expected Output

```
WgpuDevice initialized successfully!
  Adapter: AMD Radeon Graphics (RADV GFX1151)
  Backend: Vulkan
  Vendor: AMD
  Device Type: DiscreteGpu

Note: Model will be loaded on CPU (VarBuilder limitation)
Loading model as F32 (will convert to BF16 for GPU operations)
Device: Cpu, dtype: F32

Loading sharded model from index file
  [1] /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/model-00001.safetensors
  [2] /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/model-00002.safetensors
  ...
Loaded the model in 45.23s

Wgpu backend is initialized and ready for operations!
Note: Full GPU acceleration will be implemented in future versions.
Currently, inference runs on CPU with WgpuDevice available for testing.

Prompt tokens: 12 tokens
[Generated text here...]
150 tokens generated (5.32 token/s)
```

## Troubleshooting

### OpenSSL Error

If you see openssl-related build errors locally (jam):
- This is expected - jam doesn't have openssl-dev
- Build on Lyuda instead where dependencies are installed

### WGPU Not Available

If WGPU initialization fails:
```
Wgpu not available, using CPU
```

Check:
- Vulkan drivers: `vulkaninfo | head -20`
- GPU visibility: `lspci | grep VGA`
- Mesa version: `glxinfo | grep "OpenGL version"`

### Build Errors

If cargo build fails:
```bash
# Clean build cache
cargo clean

# Try building just the example
cargo build --example qwen3_omni_wgpu --release
```

## Architecture Limitations

**Current**: Model loads on CPU because VarBuilder only supports CPU/CUDA/Metal devices.

**Future**: To enable GPU model loading:
1. Add `Device::Wgpu(WgpuDevice)` variant to `candle-core`
2. Add `Storage::Wgpu(WgpuStorage)` variant
3. Update VarBuilder to handle WgpuDevice
4. Implement tensor transfer operations

## Performance Notes

- **CPU Inference**: ~5-10 tokens/sec (depending on model size)
- **Future GPU Inference**: Expected ~50-100 tokens/sec with WGPU backend
- **Memory**: Model loaded via mmap, minimal RAM usage

## Next Steps

1. Test compilation on Lyuda
2. Verify WGPU device detection
3. Run basic inference test
4. Compare performance with CPU-only example
5. Plan integration with actual GPU operations

## Related Files

- `/candle-examples/examples/qwen3_omni_wgpu/main.rs` - Example code
- `/candle-examples/examples/qwen3_omni_wgpu/README.md` - Technical details
- `/candle-wgpu/src/device.rs` - WGPU device implementation
- `/candle-wgpu/src/storage.rs` - WGPU storage implementation
