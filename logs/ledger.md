# Ledger

## 2026-01-30 04:45
Done: 6 research agents completed (RDNA3.5 shader opts, fused ops, NPU, buffer pool, BF16 native output, GPU profiling). CRITICAL FINDING: decode is 100% memory-bound (AI≈1.0 vs ridge 56.42), bandwidth utilization only 2.8%, theoretical max 219 tok/s vs current 5 tok/s (44x gap). Subgroup ops fully supported in wgpu v24+. NPU unusable on Linux. Buffer pool can give >97% reuse. 106 cast dispatches eliminable. All docs in materials/research/.
Next: Priority optimizations based on memory-bound insight: (1) Vectorized BF16 loads (vec4<u32> = 4x bandwidth), (2) Subgroup reductions (replace shared memory), (3) BF16 native output (eliminate 106 casts), (4) Buffer pool (eliminate 1571 allocs/token), (5) Fused ops (RMSNorm+Linear, QKV).

## 2026-01-29 09:06
Done: Deep research on GPU shader operation fusion for LLM inference completed. Investigated: (1) Common ML compiler fusion patterns (FlashAttention, RMSNorm+Linear, fused MLP, activation fusion), (2) llama.cpp Vulkan backend design (callback-based dequant fusion, split-K FlashAttention, matmul shader architecture), (3) WGSL constraints for fusion (workgroup limits, no cooperative matrices yet, subgroup ops via extension), (4) Cast elimination strategies (keep F32 intermediates, BF16 only at boundaries), (5) AMD RDNA 3.5 WMMA capabilities (16x16x16 tiles, 512 FLOPS/CU BF16, wave64 support). Key findings: RMSNorm+Linear fusion proven at 1.5-1.7x via mathematical reordering (Mirage approach), cast elimination can reduce 20-30% overhead, shader-level fusion preferred over command buffer batching (confirmed by previous RADV pipeline parallelism insight). Saved comprehensive report to materials/research/fused-operations.md with prioritized implementation roadmap.
Next: Implement Phase 1 fusions — cast elimination (modify matmul/layer_norm to keep F32), fused Linear+SiLU shader. Target: 5-6 tok/s improvement. Then Phase 2: RMSNorm+Linear fusion for 1.5-1.7x boost.

## 2026-01-30 03:00
Done: Tested all 3 optimizations on Lyuda individually, merged GEMV shader. Results:
- GEMV shader: 4.16-5.08 tok/s (+7-30% vs 3.90 baseline) — MERGED
- Batch dispatch: 3.47 tok/s (-11%) — NOT merged, Mutex overhead + loss of GPU pipeline parallelism
- Matmul tiling: 3.51 tok/s (-10%) — NOT merged, BM=64 tile wastes 63/64 rows for m=1 decode
- Combined GEMV+batch: 3.22 tok/s — even worse together
Key learning: wgpu/RADV benefits from frequent small submits (pipeline parallelism), not batched encoders. Bottleneck is shader execution, not submit overhead.
Next: Profile what's actually slow in the shaders. Consider: fused ops, BF16-native output, subgroup operations for RDNA3.

## 2026-01-30 01:00
Done: Implemented contiguous weight cache in Linear::new() (weight_t field) + fixed copy_strided 2D dispatch for large tensors (>65535 workgroups) + made k.transpose contiguous in attention. Result: 0.33 → 3.90 tok/s (12x), zero CPU fallback matmuls.
Next: Further optimization — batch dispatches (700 submits/token), GEMV shader for m=1, matmul tiling with shared memory. Target: 50-100 tok/s.

## 2026-01-30 00:30
Done: 5 parallel research agents completed — RDNA 3.5 optimization, NPU, non-contiguous matmul, dispatch overhead, BF16 matmul tiling. Key findings: (1) 4500/9560 matmuls fall to CPU because .t() in Linear creates non-contiguous RHS, (2) 700 queue.submit() per token, (3) no tiling/shared memory in matmul shader, (4) NPU unusable on Linux. Priority #1: make weights contiguous at load time.
Next: Implement contiguous weight cache in Linear::new(), test on Lyuda. Then batch dispatches, then GEMV shader.

## 2026-01-29 23:45
Done: Research completed on WGSL compute shader optimization for AMD RDNA 3.5 (gfx1151) BF16 ML inference. Gathered comprehensive findings on hardware specs, WMMA instructions, cache hierarchy, memory patterns, occupancy, and performance benchmarks.
Next: Apply findings to optimize candle-wgpu BF16 implementation — investigate WMMA cooperative matrix support in RADV, optimize tile sizes to 16x16 for RDNA 3, address memory bandwidth bottlenecks.

## 2026-01-29 22:00
Done: Buffer alignment fix committed (9c944ab1). All 10 shader tests pass on Lyuda (RADV/gfx1151). Inference works at 0.33 tok/s GPU vs 0.26 tok/s CPU. Profiling shows 4500/9560 matmuls fall to CPU due to non-contiguous RHS from .t() in Linear layer.
Next: Research phase — 5 parallel agents investigating bottlenecks.

## 2026-01-30 04:30
Done: Deep research completed on AMD XDNA NPU Linux support status (January 2026). Key findings: (1) Kernel driver mainlined in Linux 6.14, (2) Major gap between mainline and out-of-tree versions, (3) Firmware compatibility issues between linux-firmware and AIE stack, (4) No working llama.cpp NPU support on Linux (Windows-only), (5) IREE/MLIR-AIE promising but immature, (6) Strix Halo NPU=50 TOPS but segfaults on Linux inference tools. **Verdict: NPU remains unusable for practical ML inference on Linux (January 2026).** GPU via Vulkan/ROCm is the only working path.
Next: Continue GPU optimization path — no NPU support available.

## 2026-01-29 07:09
--- COMPACTING (auto) ---

## 2026-01-29 08:58
Done: Completed comprehensive research on GPU buffer allocation optimization for wgpu ML inference. Analyzed current allocation patterns (43 call sites, ~1571 buffers/token, 0% reuse). Researched: wgpu/VMA allocator costs, PyTorch caching allocator strategy, llama.cpp memory patterns, AMD APU UMA optimization, buffer pool patterns. Key finding: PyTorch-style two-pool allocator can reduce steady-state allocations from ~1571 to ~0-50 (>97% reuse). Document saved to materials/research/buffer-pool-memory.md with implementation roadmap.

Next: Implement Priority 1 (BufferPool in device.rs) — expected 5-15% inference speedup with 2-3x memory overhead (acceptable on 128GB UMA).
