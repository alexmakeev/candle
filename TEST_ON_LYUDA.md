# Test qwen3_omni_wgpu Example on Lyuda

## Step-by-Step Testing Guide

### 1. Push Changes from Local (jam)

```bash
cd /home/alexii/lluda/candle-16b

# Check what was created
git status

# Add all new files
git add candle-examples/examples/qwen3_omni_wgpu/
git add candle-examples/Cargo.toml
git add test_wgpu_example.sh
git add WGPU_EXAMPLE_GUIDE.md
git add WGPU_EXAMPLE_SUMMARY.md
git add TEST_ON_LYUDA.md

# Commit
git commit -m "Add qwen3_omni_wgpu example for testing WGPU backend

- Create qwen3_omni_wgpu example based on qwen3_omni_text
- Initialize WgpuDevice for GPU detection
- Load model on CPU (VarBuilder limitation)
- Add comprehensive documentation and guides
- Add build test script

Files:
- candle-examples/examples/qwen3_omni_wgpu/main.rs (372 lines)
- candle-examples/examples/qwen3_omni_wgpu/README.md
- candle-examples/Cargo.toml (add candle-wgpu dependency)
- test_wgpu_example.sh
- WGPU_EXAMPLE_GUIDE.md
- WGPU_EXAMPLE_SUMMARY.md
- TEST_ON_LYUDA.md"

# Push to remote
git push origin qwen3-omni-16b
```

### 2. Connect to Lyuda

```bash
# Primary tunnel (auto)
sshpass -p '1q2w3e' ssh -p 2233 lluda@127.0.0.1

# If primary fails, try backup
sshpass -p '1q2w3e' ssh -p 2222 lluda@127.0.0.1
```

### 3. Update Code on Lyuda

```bash
cd ~/candle-16b
git status  # Check current state
git pull origin qwen3-omni-16b
```

Expected output:
```
From https://github.com/your-repo/candle
 * branch            qwen3-omni-16b -> FETCH_HEAD
Updating abc1234..def5678
Fast-forward
 candle-examples/Cargo.toml                              |   1 +
 candle-examples/examples/qwen3_omni_wgpu/README.md      | 100 +++++++
 candle-examples/examples/qwen3_omni_wgpu/main.rs        | 372 +++++++++++++++++++++
 test_wgpu_example.sh                                    |  25 ++
 ...
```

### 4. Build the Example

```bash
# Option A: Use test script
./test_wgpu_example.sh

# Option B: Manual build
cargo build --example qwen3_omni_wgpu --release
```

Expected build time: ~5-10 minutes (first time)

### 5. Run the Example

```bash
# Short test
cargo run --example qwen3_omni_wgpu --release -- \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni \
  --prompt "What is 2+2?" \
  --sample-len 50

# Full test
cargo run --example qwen3_omni_wgpu --release -- \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni \
  --prompt "Explain what deep learning is in simple terms" \
  --sample-len 200 \
  --temperature 0.7
```

### 6. Expected Output

```
avx: false, neon: false, simd128: false, f16c: false
temp: 0.70 repeat-penalty: 1.10 repeat-last-n: 64

WgpuDevice initialized successfully!
  Adapter: AMD Radeon Graphics (RADV GFX1151)
  Backend: Vulkan
  Vendor: 4098
  Device Type: DiscreteGpu

Loading tokenizer from: /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/tokenizer.json
Loading config from: /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/config.json
Config: ThinkerConfig { ... }
Retrieved files in 0.05s

Note: Model will be loaded on CPU (VarBuilder limitation)
Future versions will support direct GPU loading
Loading model as F32 (will convert to BF16 for GPU operations)
Device: Cpu, dtype: F32

Loading sharded model from index file
  [1] /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/model-00001.safetensors
  [2] /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/model-00002.safetensors
  ...
Loading Thinker model...
Loaded the model in 45.23s

Wgpu backend is initialized and ready for operations!
Note: Full GPU acceleration will be implemented in future versions.
Currently, inference runs on CPU with WgpuDevice available for testing.

Prompt tokens: 12 tokens
What is 2+2? The answer is 4...
50 tokens generated (5.32 token/s)
```

### 7. Verification Checklist

Check the output for:

- [ ] **WgpuDevice initialized successfully** - WGPU backend working
- [ ] **Adapter: AMD Radeon Graphics** - Correct GPU detected
- [ ] **Backend: Vulkan** - Vulkan backend active
- [ ] **Device Type: DiscreteGpu** - Recognized as discrete GPU
- [ ] **Model loaded in ~45s** - Normal loading time
- [ ] **Text generated** - Inference working
- [ ] **~5-10 token/s** - Reasonable CPU performance

### 8. Test CPU Fallback

```bash
# Force CPU mode (no WGPU)
cargo run --example qwen3_omni_wgpu --release -- \
  --cpu \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni \
  --prompt "Test" \
  --sample-len 20
```

Expected:
```
Wgpu not available, using CPU
Device: Cpu, dtype: F32
```

### 9. Compare with Original Example

```bash
# Run original CPU example
cargo run --example qwen3_omni_text --release -- \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni \
  --prompt "What is 2+2?" \
  --sample-len 50

# Compare performance
# Should be similar since both run on CPU
```

## Troubleshooting

### Build Fails

```bash
# Clean and rebuild
cargo clean
cargo build --example qwen3_omni_wgpu --release
```

### WGPU Not Detected

```bash
# Check Vulkan
vulkaninfo | head -20

# Check GPU
lspci | grep VGA

# Check Mesa
glxinfo | grep "OpenGL version"
```

### Model Not Found

```bash
# Verify model path
ls -lh /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/

# Should see:
# - config.json
# - tokenizer.json
# - model-*.safetensors files
```

### Low Performance

Check:
- CPU usage: `htop`
- Memory: `free -h`
- Thermal throttling: `sensors`

## Success Criteria

Example is working if:

1. ✓ Compiles without errors
2. ✓ Detects AMD Radeon 8060S via WgpuDevice
3. ✓ Loads model successfully on CPU
4. ✓ Generates coherent text responses
5. ✓ Performance similar to qwen3_omni_text example (~5-10 tok/s)

## Next Steps After Success

1. Document actual performance numbers
2. Compare with qwen3_omni_text baseline
3. Plan Device::Wgpu integration in candle-core
4. Design tensor transfer operations
5. Implement actual GPU inference

## Results Template

```markdown
## Test Results

**Date**: YYYY-MM-DD
**Machine**: Lyuda (AMD Ryzen AI Max+ 395, Radeon 8060S)
**Branch**: qwen3-omni-16b

### Build
- Compilation: ✓/✗
- Time: X minutes

### GPU Detection
- WgpuDevice init: ✓/✗
- Adapter: [name]
- Backend: [Vulkan/Metal/DX12]

### Inference
- Model loading: ✓/✗ (X seconds)
- Text generation: ✓/✗
- Performance: X.XX tok/s

### Issues
[List any problems encountered]

### Notes
[Additional observations]
```
