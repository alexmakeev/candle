## 2026-01-28 23:45
Done: Deep research — wgpu backend best practices for ML inference
- Saved to docs/wgpu_research.md with 80+ sources
- Key findings: 16x16 tiling for matmul, BF16 via bit-shift (no native WGSL support), VK_KHR_shader_bfloat16 NOT available on RDNA 3.5 (only RDNA 4), WMMA available via ROCm but not through wgpu, 128 KB LDS per WGP, wgpu 4GB buffer limit requires split buffers, UMA zero-copy limited by wgpu MAP_WRITE|STORAGE restriction
- Burn/CubeCL is most mature wgpu ML framework, TokenHawk has hand-written WGSL transformer shaders
- llama.cpp WebGPU backend actively developed with WGSL matmul shaders

Next: Apply research to optimize wgpu backend matmul (tiling, shared memory), implement softmax/RMSNorm/RoPE shaders

## 2026-01-28 21:30
Done: Integrated wgpu backend into candle-core
- Moved wgpu_backend module from separate crate to candle-core/src/wgpu_backend/
- Added Device::Wgpu variant and all BackendDevice methods
- Added Storage::Wgpu variant with matmul (F32, BF16), binary ops, to_cpu
- Fixed buffer alignment for COPY_BUFFER_ALIGNMENT (4 bytes)
- Added wgpu cases to: binary_impl, matmul, index_select, to_device, to_vec*, display
- Created wgpu_basics.rs and wgpu_varbuilder.rs examples (both pass)

Tests (via llvmpipe software Vulkan):
- zeros, ones, from_slice, add, matmul — OK
- BF16 tensor creation and matmul — OK
- SafeTensors load to WgpuDevice — OK
- to_device CPU↔Wgpu — OK

Next: Push to Lyuda, test on real GPU (Radeon 8060S), load Qwen3-Omni

## 2026-01-28 20:15
Done: Text completion example for Qwen3-Omni BF16
- Created candle-examples/examples/qwen3_omni_text/main.rs
- Fixed config parsing: thinker_config.text_config extraction
- Fixed tensor prefix: added "thinker" to VarBuilder
- Made audio_embed and talker_head optional for text-only models
- Model loads config correctly (hidden_size=2048, 48 layers, vocab=152064)

Issue: OOM on CPU mode (66GB BF16 → 132GB F32)
- Lyuda has Vulkan/wgpu GPU, not CUDA
- Need wgpu backend integration for GPU inference

Next: Integrate candle-wgpu or find workaround for BF16 on CPU

## 2026-01-28 16:42
--- COMPACTING (auto) ---
