# WGPU Backend для ML Inference: Исследование лучших практик

> Дата: 2026-01-28
> Целевое железо: AMD Radeon 8060S (RDNA 3.5, GFX1151, 40 CU)
> Модель: Qwen3-Omni 30B MoE (BF16, ~32GB)
> Framework: Candle (Rust)

---

## 1. Compute Shader Best Practices для MatMul на wgpu

### 1.1 Оптимальный размер workgroup

- **Минимум:** 64 invocations (одна wavefront на AMD)
- **Рекомендация для 2D matmul:** `@workgroup_size(16, 16)` = 256 threads — максимум для WebGPU
- **AMD-специфика:** workgroup size должен быть кратен 64 для оптимальной загрузки wavefront'ов ([RDNA Performance Guide](https://gpuopen.com/learn/rdna-performance-guide/))
- wgpu/WebGPU ограничивает максимум invocations в workgroup до 256

### 1.2 Tiling Strategies для MatMul

**Прогрессия оптимизации** (от простого к сложному):

1. **Naive:** один thread на один элемент выхода, `@workgroup_size(8, 8)` — ~1.64 GFLOPS
2. **2D Output Tiling:** каждый thread считает tile 4x4 или 8x8 выхода — 3x ускорение при переходе с 4x4 на 8x8
3. **Shared Memory Tiling:** загрузка tiles A и B в `var<workgroup>` с `workgroupBarrier()` — значительное ускорение за счёт data reuse
4. **Double Buffering:** перекрытие загрузки следующего tile с вычислением текущего — скрытие латентности памяти

**Оптимальные размеры tile:**
- RDNA 3.5 WMMA работает с **16x16x16** фиксированными блоками
- Для software tiling в WGSL: **16x16** tile с 4x4 или 8x8 per-thread output — хороший баланс
- Burn framework использует иерархическую систему: (64, 32, 64)-Matmul с k-loop шагом 16

**Достижимая производительность:**
- WebGPU matmul kernel достигает 1+ TFLOPS на Apple M2 Pro (~17% от пиковых 6 TFLOPS) ([Optimizing a WebGPU Matmul Kernel for 1TFLOP+](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel))
- CUDA cuBLAS достигает ~75% от пика без tensor cores
- Burn CubeCL с автотюнингом приближается к LibTorch по производительности

### 1.3 Shared Memory / Workgroup Storage

```wgsl
// Пример: tiled matmul shared memory pattern
var<workgroup> tile_a: array<array<f32, TILE_SIZE>, TILE_SIZE>;
var<workgroup> tile_b: array<array<f32, TILE_SIZE>, TILE_SIZE>;

// Load tile into shared memory
tile_a[local_id.y][local_id.x] = a[global_row][k_offset + local_id.x];
tile_b[local_id.y][local_id.x] = b[k_offset + local_id.y][global_col];
workgroupBarrier();

// Compute partial sum from shared memory
for (var k = 0u; k < TILE_SIZE; k++) {
    sum += tile_a[local_id.y][k] * tile_b[k][local_id.x];
}
workgroupBarrier();
```

**Ключевые правила:**
- `workgroupBarrier()` обязателен после загрузки и перед чтением
- На RDNA 3.5: **128 KB LDS на WGP** (2 CU = 1 WGP), т.е. **64 KB на CU**
- Векторизация записи в shared memory автоматически распределяет транзакции по банкам — padding обычно не нужен

### 1.4 Memory Coalescing

- Writes в images/buffers должны идти **coalesced 256-byte блоками на wave** ([RDNA Performance Guide](https://gpuopen.com/learn/rdna-performance-guide/))
- Последовательные threads должны обращаться к последовательным адресам в памяти
- При загрузке матрицы B — transpose или использование shared memory для перестановки доступа

### 1.5 Subgroups (Wave-level Operations)

- WebGPU поддерживает subgroups начиная с Chrome 125
- Позволяют обмениваться данными между threads в wave без shared memory — ниже латентность
- AMD RDNA поддерживает DPP (Data Parallel Processing) и LDS Permute для intra-wave коммуникации
- На Intel GPU subgroups дают 2.5x ускорение, но результаты варьируются по GPU ([gpuweb/gpuweb#3950](https://github.com/gpuweb/gpuweb/issues/3950))

**Ограничение:** размер subgroup зависит от железа и компилятора (32 или 64 на AMD), что усложняет портативность

---

## 2. BF16 Support в wgpu/WGSL

### 2.1 Текущий статус

- **WGSL не имеет нативной поддержки BF16**
- WGSL поддерживает `f16` (IEEE 754 half-precision) через `enable f16;`, но это НЕ bfloat16
- BF16 рассматривался в ранних обсуждениях WGSL как "potentially exotic floating point type" для будущего
- Нет built-in функций `pack2xbf16()` / `unpack2xbf16()` — существующие `pack2x16float()` / `unpack2x16float()` работают только с IEEE f16

### 2.2 Software BF16 через битовые операции

BF16 — это верхние 16 бит f32. Конвертация тривиальна:

```wgsl
// BF16 → F32 (unpack)
fn bf16_to_f32(bf16_bits: u32) -> f32 {
    return bitcast<f32>(bf16_bits << 16u);
}

// F32 → BF16 (pack, truncation без округления)
fn f32_to_bf16(value: f32) -> u32 {
    return bitcast<u32>(value) >> 16u;
}

// F32 → BF16 (pack, с округлением к ближайшему чётному)
fn f32_to_bf16_rne(value: f32) -> u32 {
    let bits = bitcast<u32>(value);
    let rounding_bias = ((bits >> 16u) & 1u) + 0x7FFFu;
    return (bits + rounding_bias) >> 16u;
}

// Два BF16 в одном u32 (packed)
fn unpack2xbf16(packed: u32) -> vec2<f32> {
    let lo = bitcast<f32>((packed & 0xFFFFu) << 16u);
    let hi = bitcast<f32>(packed & 0xFFFF0000u);
    return vec2<f32>(lo, hi);
}
```

**Производительность:** одна shift + один bitcast — практически бесплатно. Основной overhead в том, что все вычисления идут в f32.

### 2.3 VK_KHR_shader_bfloat16 (Vulkan)

- Расширение вышло в марте 2025 с Vulkan 1.4.311 ([VK_KHR_shader_bfloat16](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_shader_bfloat16.html))
- **RDNA 3 (GFX11): НЕ поддерживается** — обнаружены проблемы с точностью BF16 операций в RADV ([Phoronix: RADV BFloat16](https://www.phoronix.com/news/RADV-Shader-BFloat16))
- **RDNA 4 (GFX12): Поддерживается** в RADV (Mesa 25.2+)
- **AMDVLK:** пока не экспонирует BF16 extension (на май 2025)
- **wgpu: не экспонирует** VK_KHR_shader_bfloat16

### 2.4 WMMA с BF16 на RDNA 3.5

Хотя VK_KHR_shader_bfloat16 не поддерживается через shader-уровень на GFX11, WMMA инструкции на RDNA 3/3.5 **нативно поддерживают BF16**:

- `V_WMMA_F32_16X16X16_BF16` — BF16 вход → F32 аккумулятор
- `V_WMMA_BF16_16X16X16_BF16` — BF16 вход → BF16 выход

Доступ через:
- HIP/ROCm — прямой доступ к WMMA intrinsics
- Vulkan — через VK_KHR_cooperative_matrix (если поддержан)
- wgpu — пока **нет доступа** к WMMA

### 2.5 Рекомендуемая стратегия для нашего проекта

1. **Хранение весов:** BF16 packed по 2 значения в u32 в storage buffers
2. **Compute shaders:** unpack BF16→F32 через bit shift, вычисления в F32, pack обратно
3. **Будущее:** мониторить wgpu native extensions для Vulkan WMMA/BF16 passthrough
4. **Альтернатива:** если нужна WMMA производительность — рассмотреть direct Vulkan compute через ash/vulkano вместо wgpu

---

## 3. Softmax, LayerNorm, RoPE шейдеры

### 3.1 Numerically Stable Softmax

**Алгоритм (два прохода + один проход вычисления):**

```wgsl
// Проход 1: найти max по строке
var max_val = -1e38;  // large negative
for (var i = 0u; i < N; i++) {
    max_val = max(max_val, input[row * N + i]);
}

// Проход 2: sum of exp(x - max)
var sum_exp = 0.0;
for (var i = 0u; i < N; i++) {
    sum_exp += exp(input[row * N + i] - max_val);
}

// Проход 3: нормализация
for (var i = 0u; i < N; i++) {
    output[row * N + i] = exp(input[row * N + i] - max_val) / sum_exp;
}
```

**Оптимизации для GPU:**
- **Parallel reduction** для max и sum внутри workgroup с shared memory
- **Fused softmax:** все три прохода в одном kernel, данные остаются в registers/shared memory
- **Online softmax** (алгоритм Milakov-Gimelshein): один проход, обновляет max и sum одновременно — меньше чтений из глобальной памяти
- Подвычитание max предотвращает overflow в `exp()` — обязательно для f32 и критически важно для f16/bf16

**Источники:**
- [Numerically Stable Softmax](https://blester125.com/blog/softmax.html)
- [GPU kernel optimization: Softmax](https://medium.com/@hugo.rosenkranz/gpu-kernel-optimization-softmax-part-1-8ff80766cc95)
- [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

### 3.2 LayerNorm (RMSNorm)

Qwen3 использует RMSNorm (без mean subtraction):

```wgsl
// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
// Проход 1: sum of squares
var sum_sq = 0.0;
for (var i = 0u; i < hidden_size; i++) {
    let v = input[offset + i];
    sum_sq += v * v;
}
let rms = sqrt(sum_sq / f32(hidden_size) + eps);

// Проход 2: нормализация
for (var i = 0u; i < hidden_size; i++) {
    output[offset + i] = (input[offset + i] / rms) * weight[i];
}
```

**Оптимизации:**
- Parallel reduction для sum of squares в shared memory
- `inverseSqrt()` вместо `1.0 / sqrt()` для лучшей производительности
- Fused: RMSNorm + следующая операция (projection) в одном kernel

### 3.3 RoPE (Rotary Position Embedding)

```wgsl
// RoPE применяет вращение к парам элементов
// theta_i = base^(-2i/d) * position
fn apply_rope(x: ptr<function, array<f32, D>>, pos: u32, base: f32) {
    for (var i = 0u; i < D / 2u; i++) {
        let theta = pow(base, -f32(2u * i) / f32(D)) * f32(pos);
        let cos_t = cos(theta);
        let sin_t = sin(theta);
        let x0 = (*x)[2u * i];
        let x1 = (*x)[2u * i + 1u];
        (*x)[2u * i]     = x0 * cos_t - x1 * sin_t;
        (*x)[2u * i + 1u] = x0 * sin_t + x1 * cos_t;
    }
}
```

**Оптимизации:**
- Предвычисление cos/sin таблиц для всех позиций — загрузка из buffer вместо вычисления в runtime
- Каждый thread обрабатывает одну пару (x0, x1) — хорошо параллелизуется
- Fused с attention QK projection

### 3.4 Fused Attention (Flash Attention стиль)

Для Qwen3-Omni с MoE — attention является bottleneck:
- Fused Q*K^T → softmax → V в одном kernel
- Tiled attention: загрузка блоков Q, K, V в shared memory
- Online softmax позволяет вычислять attention без полной материализации attention matrix

**Референсные реализации:**
- [TokenHawk](https://github.com/kayvr/token-hawk) — hand-written WGSL шейдеры для LLaMA inference
- [wgml](https://github.com/wgmath/wgml) — Rust библиотеки с WebGPU шейдерами для LLM inference
- llama.cpp WebGPU backend — WGSL шейдеры для matmul, unary ops ([PR #17031](https://github.com/ggml-org/llama.cpp/pull/17031))

---

## 4. Memory Management для больших моделей

### 4.1 wgpu Buffer Size Limits

**Критическая проблема:** wgpu определяет `max_storage_buffer_binding_size` как `u32`, что ограничивает один buffer до ~4 GB. На практике лимит может быть **~2 GB** (NVIDIA H100 через wgpu показывает 2147483647 байт).

- [Issue #2337: Cannot allocate buffers over 4GiB](https://github.com/gfx-rs/wgpu/issues/2337)
- [Issue #8105: Allow storage buffer limits beyond i32 max](https://github.com/gfx-rs/wgpu/issues/8105)
- Vulkan native позволяет буферы до размера всей доступной памяти (`VkDeviceSize = uint64_t`)
- Исправление (переход на u64) запрошено с 2021 года, но не решено на 2025

### 4.2 Стратегия для 32GB модели

**Обязательно:** разбиение на множество буферов < 2-4 GB.

**Вариант 1: Split Buffers с Binding Arrays**
```rust
// Создание множества буферов по chunk_size
let chunk_size = 2 * 1024 * 1024 * 1024; // 2 GB
let num_chunks = (total_size + chunk_size - 1) / chunk_size;
let buffers: Vec<Buffer> = (0..num_chunks)
    .map(|i| device.create_buffer(&BufferDescriptor {
        size: min(chunk_size, total_size - i * chunk_size),
        usage: BufferUsages::STORAGE,
        ..
    }))
    .collect();
```

Требует:
- `STORAGE_RESOURCE_BINDING_ARRAY` и `BUFFER_BINDING_ARRAY` features в wgpu
- Шейдер должен обрабатывать индексацию через массив буферов
- [Discussion #6130: Big big big buffers](https://github.com/gfx-rs/wgpu/discussions/6130)

**Вариант 2: Per-Layer Buffers**
- Один buffer на каждый слой/матрицу весов (обычно < 1 GB каждый)
- Проще в реализации, не требует binding arrays
- Для Qwen3-Omni 30B: ~100+ отдельных weight tensors

**Вариант 3: Lazy Loading + Buffer Pool**
- Загружать только активные expert'ы в MoE (из 128 expert'ов активны ~8)
- Pool переиспользуемых буферов фиксированного размера
- Swap между CPU и GPU по необходимости

### 4.3 UMA на Strix Halo

**Ключевое преимущество Strix Halo:** Unified Memory Architecture — CPU и GPU разделяют одну и ту же физическую память (до 128 GB LPDDR5X).

**Конфигурация памяти:**
- **GART (Graphics Address Remapping Table):** фиксированный резерв в BIOS для GPU — рекомендуется выставить минимум (512 MB)
- **GTT (Graphics Translation Table):** динамически аллоцируемая память — основной механизм для ML
- Без правильной настройки kernel parameters контейнеры ограничены 2-8 GB ([Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes))

**Vulkan UMA:**
- Vulkan на Strix Halo видит memory type с флагами `HOST_VISIBLE | DEVICE_LOCAL` одновременно
- Это позволяет zero-copy: CPU пишет напрямую в GPU-доступный буфер без staging
- RADV распознаёт UMA: "AMD Radeon Graphics (RADV GFX1151) | uma: 1"
- **AMDVLK ограничен 2 GB на один буфер**, RADV — нет

**Ограничение wgpu:**
- wgpu не позволяет `MAP_WRITE | STORAGE` на одном буфере (WebGPU spec limitation)
- Staging buffer всё ещё нужен, что добавляет один лишний copy на UMA системе
- Для zero-copy нужен прямой Vulkan доступ (ash/vulkano)

### 4.4 Bandwidth

- Реальная bandwidth: **~212 GB/s** (DDR5-8000, 256-bit bus, теоретически 256 GB/s)
- Для BF16 30B модели при inference: bandwidth-bound операции доминируют
- Memory-bound операции (embedding lookup, pointwise ops) vs compute-bound (matmul)

### 4.5 Burn Framework: Memory Management

Burn использует **chunk/slice модель:**
- **Chunk** — фиксированный регион памяти (GPU buffer)
- **Slice** — часть chunk'а определённая offset и size
- Переиспользование аллоцированной памяти через memory pool
- ([Creating High Performance Backends with Burn-Compute](https://burn.dev/blog/creating-high-performance-asynchronous-backends-with-burn-compute/))

---

## 5. Существующие проекты wgpu для ML

### 5.1 Burn (tracel-ai/burn)

**Статус:** Наиболее зрелый Rust ML framework с wgpu backend.

- **Backend:** wgpu через CubeCL (с 2024). До этого — raw WGSL templates
- **CubeCL:** Rust proc-macro `#[cube]` → генерация GPU кода для CUDA, ROCm, Vulkan, Metal, WebGPU
- **MatMul:** Иерархическая система (Batch → Global → Stage → Tile), JIT компиляция, autotune
- **Fusion:** Автоматическое слияние element-wise операций в один kernel (до 78x ускорение для GELU)
- **Производительность:** Matching или превосходит LibTorch на многих бенчмарках
- **WMMA:** Через CubeCL доступны WMMA на CUDA; на Vulkan ограничены line size = 4

**Источники:**
- [Burn GitHub](https://github.com/tracel-ai/burn)
- [SOTA Multiplatform MatMul](https://burn.dev/blog/sota-multiplatform-matmul/)
- [CubeCL](https://github.com/tracel-ai/cubecl)

### 5.2 Wonnx (webonnx/wonnx)

**Статус:** WebGPU ONNX runtime, 100% Rust.

- Преимущества: простой код, лёгкий для контрибуции
- Ограничения: нет int8 inference, нет 64-bit integers, MatMul только для float
- Не преследует цель training
- Хороший reference для понимания wgpu compute pipeline

**Источник:** [Wonnx GitHub](https://github.com/webonnx/wonnx)

### 5.3 webgpu-torch (praeclarum/webgpu-torch)

**Статус:** TypeScript, PyTorch-like API для WebGPU.

- Только float32
- Autograd поддержка (частичная)
- Полезен как reference для WebGPU kernel patterns

**Источник:** [webgpu-torch GitHub](https://github.com/praeclarum/webgpu-torch), [Blog Post](https://praeclarum.org/2023/05/19/webgpu-torch.html)

### 5.4 WebLLM (mlc-ai/web-llm)

**Статус:** Наиболее производительный browser-based LLM inference.

- Использует TVM-compiled WebGPU kernels (не hand-written WGSL)
- ~80% от native производительности на некоторых задачах
- WeInfer (2025) достигает 3.76x ускорения над WebLLM

**Источники:**
- [WebLLM GitHub](https://github.com/mlc-ai/web-llm)
- [WebLLM Paper](https://arxiv.org/html/2412.15803v1)
- [WeInfer (ACM 2025)](https://dl.acm.org/doi/10.1145/3696410.3714553)

### 5.5 TokenHawk (kayvr/token-hawk)

**Статус:** Hand-written WGSL шейдеры для LLaMA inference.

- Только llama 7B-f16
- Наиболее релевантный reference для hand-written WGSL transformer shaders
- Используется Dawn (Google C++ WebGPU implementation)
- Реализует: matmul, softmax, layernorm, RoPE, attention

**Источник:** [TokenHawk GitHub](https://github.com/kayvr/token-hawk)

### 5.6 wgml (wgmath/wgml)

**Статус:** Rust библиотеки для LLM inference через WebGPU.

- Cross-platform, web + native
- WebGPU шейдеры для LLM kernels

**Источник:** [wgml GitHub](https://github.com/wgmath/wgml)

### 5.7 llama.cpp WebGPU Backend

**Статус:** Активная разработка (2025-2026).

- WGSL шейдеры для matmul (register tiling + subgroup matrices)
- Unary ops: softplus, expm1, floor, ceil (f32/f16)
- На Llama 1B F16: ~1014 t/s pp512, ~28.7 t/s tg128 (vs Metal: ~1368/~36)
- Используется Dawn как WebGPU implementation

**Источники:**
- [Vulkan Backend DeepWiki](https://deepwiki.com/ggml-org/llama.cpp/4.4-vulkan-backend)
- [WebGPU matmul PR](https://github.com/ggml-org/llama.cpp/pull/17031)

### 5.8 Сводная таблица

| Проект | Язык | Maturity | MatMul | Transformer Ops | BF16 |
|--------|------|----------|--------|-----------------|------|
| **Burn/CubeCL** | Rust | Высокая | SOTA (иерархический, autotune) | Да | Через CubeCL |
| **Wonnx** | Rust | Средняя | Базовый | Частично (ONNX) | Нет |
| **TokenHawk** | C++ | Низкая | Hand-tuned WGSL | Да (LLaMA) | Нет (f16) |
| **wgml** | Rust | Ранняя | Да | LLM ops | Неизвестно |
| **llama.cpp WebGPU** | C++ | Ранняя | Register tiling + subgroups | В разработке | Нет |
| **WebLLM** | TypeScript | Высокая | TVM-compiled | Полный | Через TVM |

---

## 6. AMD RDNA 3.5 (GFX1151) Специфика

### 6.1 Ключевые характеристики Radeon 8060S

| Параметр | Значение |
|----------|----------|
| Архитектура | RDNA 3.5 (GFX1151) |
| Compute Units | 40 |
| WGP (Workgroup Processors) | 20 |
| Stream Processors | 2560 |
| Base Clock | 1295 MHz |
| Boost Clock | 2900 MHz |
| Память | Shared LPDDR5X (до 128 GB) |
| Memory Bus | 256-bit |
| Bandwidth | ~212 GB/s (реальная), 256 GB/s (теоретическая) |
| FP32 TFLOPS | ~14.8 |
| FP16/BF16 TFLOPS (peak) | ~59.4 (требует WMMA или wave32 VOPD) |
| FP16/BF16 TFLOPS (реальная, с hipBLASLt) | ~36.9 |
| Процесс | TSMC 4nm |
| L2 Cache | 8 MB |
| LDS на WGP | 128 KB |
| VGPR | 192 KB (per CU) |

**Источники:**
- [ROCm GPU Specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [Strix Halo LLM Optimization](https://www.hardware-corner.net/strix-halo-llm-optimization/)
- [Chips and Cheese: RDNA 3.5 LLVM Changes](https://chipsandcheese.com/p/amd-rdna-3-5s-llvm-changes)
- [NotebookCheck Radeon 8060S](https://www.notebookcheck.net/AMD-Radeon-8060S-Benchmarks-and-Specs.942049.0.html)

### 6.2 WMMA (Wave Matrix Multiply Accumulate)

RDNA 3.5 наследует WMMA от RDNA 3 (GFX11 ISA):

**Поддерживаемые инструкции:**
| Инструкция | Вход | Выход | Tile Size |
|-----------|------|-------|-----------|
| `V_WMMA_F32_16X16X16_F16` | F16 | F32 | 16x16x16 |
| `V_WMMA_F32_16X16X16_BF16` | BF16 | F32 | 16x16x16 |
| `V_WMMA_F16_16X16X16_F16` | F16 | F16 | 16x16x16 |
| `V_WMMA_BF16_16X16X16_BF16` | BF16 | BF16 | 16x16x16 |
| `V_WMMA_I32_16X16X16_IU8` | INT8 | I32 | 16x16x16 |
| `V_WMMA_I32_16X16X16_IU4` | INT4 | I32 | 16x16x16 |

**Доступ:**
- Через HIP/ROCm: непосредственно (rocWMMA)
- Через Vulkan: VK_KHR_cooperative_matrix (статус поддержки на RDNA 3.5 — проверить)
- Через wgpu: **не поддерживается** (WebGPU subgroup_matrix feature в разработке)

**Источники:**
- [How to accelerate AI on RDNA 3 using WMMA](https://gpuopen.com/learn/wmma_on_rdna3/)
- [RDNA 3.5 ISA Reference Guide (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna35_instruction_set_architecture.pdf)

### 6.3 DP4a (Dot Product 4x INT8 + Accumulate)

- RDNA 3.5 поддерживает DP4a через WMMA IU8 инструкции
- Vulkan 1.3 core: `VK_KHR_shader_integer_dot_product`
- WebGPU: Chrome 123+ поддерживает DP4a built-in ([Proposal: gpuweb#2677](https://github.com/gpuweb/gpuweb/issues/2677))
- Для INT8 quantized inference: 2 u32 содержат 8 INT8 элементов, dot product в одной инструкции

### 6.4 Wave64 vs Wave32

- RDNA 3.5 поддерживает оба режима; **драйвер выбирает автоматически**
- **Рекомендация:** проектировать шейдеры для wave64, выделять workgroups кратными 64
- Wave32: позволяет VOPD dual-issue (две операции за цикл)
- Wave64: на RDNA 2+ компилятор может использовать dual-issue "бесплатно"
- GCN выполняет полную wave64 даже при неактивных threads; RDNA может пропустить wave32-half если все threads неактивны

### 6.5 LDS (Local Data Share)

- **128 KB на WGP** (Workgroup Processor = 2 CU)
- LDS bandwidth: ~5 TB/s (для iGPU — впечатляющее значение)
- LDS latency на RDNA 3.5 значительно улучшена по сравнению с предыдущими поколениями
- Threads в одном wave могут использовать DPP/LDS Permute для ещё более быстрой коммуникации

### 6.6 Оптимальный параллелизм

- 40 CU = 20 WGP
- При wave64: до 40 активных wavefront'ов одновременно (минимум)
- С occupancy > 1 wave/CU: значительно больше
- Для compute shader: запускать минимум 40 * 64 = 2560 threads для полной загрузки
- Оптимально: 5-10x больше wavefront'ов чем CU для скрытия латентности

### 6.7 RDNA 3.5 улучшения vs RDNA 3

- 2x per-cycle texel output в compute units (компенсирует низкие клоки в low-power)
- Double rate для rich vector instructions (interpolation, comparison)
- Scalar FP ops добавлены (включая FP16 скалярные)
- `s_singleuse_vdst` — hint для register cache management
- 192 KB VGPR per CU (vs 128 KB на GFX1150/low-end GFX11)
- Улучшенное сжатие и batch processing для LPDDR5X

---

## 7. Архитектурные рекомендации

### 7.1 Выбор API

| Вариант | Pros | Cons |
|---------|------|------|
| **wgpu (Vulkan)** | Портативность, Rust-native, безопасность | Нет WMMA, BF16 через software, 4GB buffer limit |
| **Vulkan direct (ash)** | Полный доступ к extensions, WMMA, BF16, большие буферы | Больше boilerplate, ручное управление |
| **ROCm/HIP** | Лучшая производительность на AMD, WMMA | Только AMD, нет портативности |
| **CubeCL (Burn)** | Rust-native, многоплатформенный, autotune | Зависимость от Burn ecosystem |

**Рекомендация:** wgpu как основной backend с возможностью fallback на ash/Vulkan для критических kernels (matmul с WMMA). Candle уже Rust-native, интеграция с wgpu естественна.

### 7.2 Memory Layout

- Веса модели: BF16 packed в `Vec<u32>` (2 значения на u32), разбитые по слоям
- KV-Cache: F32 буферы (динамический размер, переиспользование через pool)
- Activations: F32 промежуточные буферы с pre-allocated pool
- MoE routing: отдельные маленькие буферы для gate logits

### 7.3 Kernel Приоритизация

1. **MatMul** — 80%+ compute time, максимальная оптимизация (tiled, shared memory, subgroups)
2. **Softmax** — fused с attention, numerically stable
3. **RMSNorm** — fused с следующей linear проекцией где возможно
4. **RoPE** — pre-computed cos/sin tables
5. **MoE Routing** — top-k selection, scatter/gather для expert dispatch

### 7.4 MoE-специфичные соображения

Qwen3-Omni 30B MoE: из 128 expert'ов активны ~8 на token.

- **Expert weights:** загружать все 128 expert'ов один раз, хранить в GPU memory (если хватает)
- С 128 GB UMA на Strix Halo: ~32 GB для весов вмещается с запасом
- **Expert dispatch:** gather/scatter operations — memory bandwidth bound
- **Batching:** группировать tokens по expert'ам для эффективного matmul

---

## 8. Ссылки и источники

### Matmul & Compute Shaders
- [Optimizing a WebGPU Matmul Kernel for 1TFLOP+](https://www.nuss-and-bolts.com/p/optimizing-a-webgpu-matmul-kernel)
- [Optimizing a Rust GPU matmul kernel](https://rust-gpu.github.io/blog/optimizing-matmul/)
- [Burn: SOTA Multiplatform MatMul](https://burn.dev/blog/sota-multiplatform-matmul/)
- [RDNA Performance Guide](https://gpuopen.com/learn/rdna-performance-guide/)

### BF16 & Data Types
- [VK_KHR_shader_bfloat16](https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_shader_bfloat16.html)
- [RADV BFloat16 Support](https://www.phoronix.com/news/RADV-Shader-BFloat16)
- [WebGPU Pack/Unpack Functions](https://webgpu.rocks/wgsl/functions/packing/)

### AMD Architecture
- [RDNA 3.5 ISA Reference Guide (PDF)](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna35_instruction_set_architecture.pdf)
- [WMMA on RDNA 3](https://gpuopen.com/learn/wmma_on_rdna3/)
- [Chips and Cheese: RDNA 3.5 LLVM Changes](https://chipsandcheese.com/p/amd-rdna-3-5s-llvm-changes)
- [Chips and Cheese: Radeon 890M (RDNA 3.5)](https://chipsandcheese.com/p/amds-radeon-890m-strix-points-bigger-igpu)
- [Strix Halo LLM Performance](https://llm-tracker.info/AMD-Strix-Halo-(Ryzen-AI-Max+-395)-GPU-Performance)
- [Strix Halo LLM Optimization Guide](https://www.hardware-corner.net/strix-halo-llm-optimization/)

### wgpu & Buffer Management
- [wgpu Issue #2337: >4GiB Buffers](https://github.com/gfx-rs/wgpu/issues/2337)
- [wgpu Discussion #6130: Big Buffers](https://github.com/gfx-rs/wgpu/discussions/6130)
- [Vulkan Memory Allocator: Usage Patterns](https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html)

### Frameworks & Projects
- [Burn Framework](https://github.com/tracel-ai/burn)
- [CubeCL](https://github.com/tracel-ai/cubecl)
- [Wonnx](https://github.com/webonnx/wonnx)
- [TokenHawk](https://github.com/kayvr/token-hawk)
- [wgml](https://github.com/wgmath/wgml)
- [WebLLM](https://github.com/mlc-ai/web-llm)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Strix Halo Toolboxes](https://github.com/kyuz0/amd-strix-halo-toolboxes)

### Softmax & Transformer Ops
- [Numerically Stable Softmax](https://blester125.com/blog/softmax.html)
- [Triton Fused Softmax Tutorial](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
- [GPU Kernel Optimization: Softmax](https://medium.com/@hugo.rosenkranz/gpu-kernel-optimization-softmax-part-1-8ff80766cc95)
- [WeInfer: WebGPU LLM Inference (ACM 2025)](https://dl.acm.org/doi/10.1145/3696410.3714553)
