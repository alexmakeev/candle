## 2026-01-29 01:30
Done: Тестирование inference на Qwen3-0.6B (1.5GB, маленькая модель для быстрой отладки)
- Модель загружается за 245ms на wgpu
- Добавлен qwen3_wgpu пример для Qwen3ForCausalLM
- Пропагация wgpu feature в candle-nn
- Добавлены Wgpu ветки в storage.rs для всех CustomOp1/2/3
- rms_norm: добавлен slow fallback через базовые тензорные ops
- rope: добавлен slow fallback через rope_slow
- BF16 matmul: расширен для batched (loop по batch dim с buffer offsets)
- Ошибка alignment: min_storage_buffer_offset_alignment=256, batch stride может быть 32

Decision: НИКАКИХ CPU fallback'ов! Всё на шейдерах.
- Убрать автоматические CPU roundtrip'ы в CustomOp1/2/3
- Каждая операция должна иметь нативный WGSL шейдер
- CPU режим — только по явному флагу, не автоматически
- Цель: 100% GPU inference через шейдеры
- NPU рассмотрим после GPU версии

Current fix: переделываю batched BF16 matmul — batch dimension через global_id.z в шейдере (не buffer offsets)

Ошибки пройденные в этой сессии:
1. ✅ wgpu buffer 256MB limit → adapter limits
2. ✅ OOM при загрузке → streaming mmap + madvise
3. ✅ GTT exhaustion → rotary first + scoped VarBuilder
4. ✅ device mismatch copy2d → Wgpu dispatch arms
5. ✅ rms_norm CustomOp2 → slow path (базовые tensor ops)
6. ✅ BF16 CPU matmul unsupported → F32 fallback (ВРЕМЕННО, будет заменён на шейдер)
7. 🔄 batched matmul alignment → batch dim в шейдере (в работе)

Next:
- Batched BF16 matmul shader с global_id.z
- Нативные WGSL шейдеры для: rms_norm_bf16, softmax_bf16, rope_bf16
- Убрать все CPU fallback'и
- Довести inference до генерации текста

## 2026-01-29 00:20
Done: Fixed wgpu buffer size limit, model loading progress
- Fixed `wgpu::Limits::default()` (256MB max) → request adapter limits (2GB max on RADV)
- Embedding layer (594MB) now loads successfully in 568ms
- Added verbose logging to Thinker::new() — layer-by-layer progress
- Model loaded 41/48 layers before OOM reboot (~112s)
- Root cause: double allocation — mmap (62GB system RAM) + GPU buffers (64GB VRAM) = 126GB > available
- Closed beads 3,4,5 (shaders already implemented)
- Actual model config: decoder_sparse_step=1 (ALL layers MoE), moe_intermediate_size=768, Thinker ~30B params ~60GB BF16

Decision: User setting BIOS VRAM to 96GB (from 64GB). With 96GB VRAM, model (60GB) fits with 36GB headroom for activations. System RAM becomes 32GB — enough for temporary mmap pages.

Next: After reboot with 96GB VRAM, re-run model loading. If loads OK, proceed to text generation (bead 7).

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

## 2026-01-28 18:33
--- COMPACTING (auto) ---

## 2026-01-28 19:54
--- COMPACTING (auto) ---
