# Ledger

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

## 2026-01-29 07:09
--- COMPACTING (auto) ---
