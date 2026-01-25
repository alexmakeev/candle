# TTS/Vocoder Debugging Guide — Qwen3-Omni Q8_0

## Контекст проблемы

- **Модель**: Qwen3-Omni (Thinker 30B MoE + Talker 3B + Code2Wav vocoder)
- **Квантизация**: Q8_0 (8-bit, 34 bytes per 32 elements)
- **Симптом**: Текстовая генерация работает ("Столица Франции?" → "Париж"), но аудио — шум
- **Pipeline**: Text → Thinker → Talker → Code2Wav → WAV

---

## Часть 1: Возможные причины шума

### 1.1 Проблемы с токенами и индексами кодбуков

| Симптом | Вероятная причина |
|---------|-------------------|
| Полный шум | Неправильный offset/смещение токенов |
| Деградирующее качество | Codebook collapse в RVQ |
| Спорадический шум | Index out of range |

**Детали:**
- Формула: `flat_index = token + cb * 4096` для 8 кодбуков
- Vocab size = 32768 = 8 × 4096
- Если offset считается неверно — embedding lookup даёт мусор

### 1.2 Квантизация Q8_0

| Симптом | Вероятная причина |
|---------|-------------------|
| Артефакты/дисторшн | Vocoder чувствителен к квантизации |
| Полный шум | Критические слои потеряли точность |
| Разное качество | Рекуантизация вместо квантизации с F32 |

**Важно:**
- Vocoder более чувствителен к квантизации, чем акустическая модель
- В BitTTS специально исключали последний слой вокодера из квантизации
- Q8_0 без imatrix может быть недостаточно для конволюционных слоёв

### 1.3 Несовпадение параметров (mismatch)

| Параметр | Ожидаемое значение |
|----------|-------------------|
| Sample rate | 24000 Hz |
| Codebook size | 4096 per codebook |
| Num codebooks | 8 (для Code2Wav) |
| Embedding dim | 1024 |
| Vocab size | 32768 |

### 1.4 Архитектурные проблемы

- **Multi-codebook ordering**: Первый кодбук = грубая структура, последующие = детали
- **Talker outputs 15 codebooks**, но Code2Wav ожидает только 8 → использовать первые 8
- **Activation function**: Убедиться, что activation совпадает (Snake/ELU/etc)

---

## Часть 2: Пошаговый план отладки

### Шаг 1: Проверить codec tokens

```rust
// Rust/Candle версия
let tokens = talker_output; // shape: [batch, seq, num_codebooks=8]

for cb in 0..8 {
    let cb_tokens = tokens.i((.., .., cb))?;
    let min_t = cb_tokens.min(0)?.to_scalar::<i64>()?;
    let max_t = cb_tokens.max(0)?.to_scalar::<i64>()?;
    println!("Codebook {}: min={}, max={}", cb, min_t, max_t);
    // Ожидается: 0 <= token < 4096

    if max_t >= 4096 || min_t < 0 {
        println!("WARNING: Codebook {} out of range!", cb);
    }
}
```

**Что проверить:**
- [ ] Все токены в диапазоне [0, 4096)
- [ ] Уникальных токенов достаточно много (не 5-8, а сотни)
- [ ] Распределение не вырождено в константу

### Шаг 2: Проверить embedding lookup

```rust
// Проверить формулу offset
for cb in 0..8 {
    for &token in sample_tokens[cb].iter() {
        let flat_idx = token + cb * 4096;
        assert!(flat_idx < 32768, "Out of range: {}", flat_idx);
    }
}

// Проверить embedding weights
let emb = &model.code2wav.code_embedding;
let weights = emb.embeddings();
let mean = weights.mean_all()?.to_scalar::<f32>()?;
let var = weights.var(0)?.mean_all()?.to_scalar::<f32>()?;
println!("Embedding stats: mean={:.4}, std={:.4}", mean, var.sqrt());
// Если std ≈ 0 — веса не загрузились
```

### Шаг 3: Изолировать проблему

**Тест 1: Нулевые токены (должна быть тишина)**
```rust
let silence_tokens = Tensor::zeros((1, 100, 8), DType::I64, device)?;
let wav_silence = code2wav.forward(&silence_tokens)?;
// Ожидается: wav close to zero или минимальный шум
```

**Тест 2: Повторяющийся паттерн**
```rust
// Один и тот же токен повторяется
let pattern = Tensor::full(512_i64, (1, 100, 8), device)?;
let wav_pattern = code2wav.forward(&pattern)?;
// Ожидается: периодический сигнал, не хаотичный шум
```

**Тест 3: Референсные токены (если есть)**
```rust
// Загрузить токены из оригинальной реализации HuggingFace
let ref_tokens = load_reference("ref_tokens.safetensors")?;
let wav_ref = code2wav.forward(&ref_tokens)?;
// Если работает — проблема в Talker
// Если шум — проблема в Code2Wav или загрузке весов
```

### Шаг 4: Проверить квантизацию

```bash
# Сравнить с F32 версией на том же input
# Если F32 работает, а Q8_0 нет — проблема в квантизации

# Проверить критические слои:
# - code2wav.code_embedding (должен быть F32 или высокая точность)
# - Последние conv слои decoder
```

**Слои чувствительные к квантизации:**
1. `code2wav.code_embedding.weight` — embedding lookup
2. `code2wav.decoder.*.conv.weight` — финальные конволюции
3. `code2wav.upsample.*.conv.weight` — upsampling слои

### Шаг 5: Проверить audio output

```rust
let wav = code2wav.forward(&tokens)?;

// Проверить диапазон
let min_val = wav.min(0)?.to_scalar::<f32>()?;
let max_val = wav.max(0)?.to_scalar::<f32>()?;
println!("WAV range: [{:.4}, {:.4}]", min_val, max_val);
// Ожидается: [-1.0, 1.0]

// Проверить sample rate
let sample_rate = 24000; // или из config
let duration = wav.dims()[1] as f32 / sample_rate as f32;
println!("Duration: {:.2}s", duration);
```

### Шаг 6: Визуализация (Python)

```python
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Загрузить WAV
wav, sr = librosa.load("output.wav", sr=None)

# Спектрограмма
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
D = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr)
plt.title('Spectrogram')
plt.colorbar()
# Шум = равномерное распределение энергии
# Речь = формантная структура, гармоники

plt.subplot(3, 1, 2)
plt.plot(wav[:2000])
plt.title('Waveform (first 2000 samples)')
# Шум = хаотичный паттерн
# Речь = периодичность

plt.subplot(3, 1, 3)
# Mel-spectrogram
mel = librosa.feature.melspectrogram(y=wav, sr=sr)
librosa.display.specshow(librosa.power_to_db(mel), y_axis='mel', x_axis='time')
plt.title('Mel Spectrogram')
plt.colorbar()

plt.tight_layout()
plt.savefig('audio_analysis.png')
```

### Шаг 7: Сравнение слой за слоем

```python
# Если есть доступ к HuggingFace реализации
from transformers import Qwen2AudioForConditionalGeneration

orig_model = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen3-Omni")

# Сравнить embedding output
orig_emb = orig_model.code2wav.code_embedding(tokens)
my_emb = my_model.get_embeddings(tokens)
diff = (orig_emb - my_emb).abs().max()
print(f"Max embedding diff: {diff}")
# Если diff > 0.01 — проблема в загрузке весов или квантизации

# Добавить hooks для сравнения каждого слоя
```

---

## Часть 3: Специфика Qwen3-Omni

### Архитектура Code2Wav

```
RVQ Codes [batch, seq_len, num_codebooks=8]
    │
    ▼
Code Embedding [32768, 1024] → sum по codebooks
    │
    ▼
Pre-Transformer (8 layers, sliding-window attention)
    │
    ▼
Upsampling (2 ConvNeXt blocks, 2x each = 4x total)
    │
    ▼
Decoder (HiFi-GAN style, multi-stage upsampling)
    │
    ▼
Waveform [batch, 1, samples] @ 24kHz
```

### Talker → Code2Wav интерфейс

**Talker выдаёт:**
- 15 codebooks (shape: [batch, seq, 15])
- Токены в диапазоне [0, 4095]

**Code2Wav ожидает:**
- 8 codebooks (первые 8 из Talker)
- Flat tokens с offset: `token + cb * 4096`
- Итого vocab_size = 32768

### Ключевые тензоры в GGUF

```
code2wav.code_embedding.weight      [1024, 32768]  — КРИТИЧНО, должен быть точным
code2wav.pre_transformer.layers.N.* — 8 слоёв transformer
code2wav.upsample.N.*               — 2 ConvNeXt блока
code2wav.decoder.N.*                — HiFi-GAN decoder
```

---

## Часть 4: Чеклист для быстрой диагностики

### Перед инференсом
- [ ] Все гиперпараметры совпадают (sample_rate, codebook_size, etc)
- [ ] Токены в правильном диапазоне
- [ ] Embedding weights загружены (mean != 0, std != 0)

### При полном шуме
- [ ] Проверить offset формулу: `token + cb * 4096`
- [ ] Проверить порядок codebooks (первый = грубая структура)
- [ ] Тест с нулевыми токенами (должна быть тишина)

### При частичных артефактах
- [ ] Проверить квантизацию критических слоёв
- [ ] Сравнить с F32 версией
- [ ] Визуализировать спектрограмму

### При проблемах с длительностью
- [ ] Проверить stop token detection
- [ ] Проверить max_length ограничения
- [ ] Соотношение input tokens / output samples

---

## Часть 5: Известные проблемы и решения

### Проблема: Мало уникальных токенов (5-8 вместо сотен)

**Причина**: Talker не сходится или генерирует константу
**Решение**: Проверить hidden_states из Thinker, убедиться что они варьируются

### Проблема: Токены выходят за диапазон

**Причина**: Неправильный vocab_size в Talker config
**Решение**: Убедиться что `talker.num_tokens = 4096` для каждого codebook

### Проблема: Работает на F32, шум на Q8_0

**Причина**: Embedding или conv слои потеряли точность
**Решение**:
1. Использовать Q8_0 с imatrix
2. Или оставить критические слои в F32

### Проблема: Talker генерирует 15 codebooks, Code2Wav ожидает 8

**Причина**: Разные версии модели
**Решение**: Использовать первые 8 codebooks: `tokens[:, :, :8]`

---

## Источники

- [Qwen3-Omni Technical Report](https://arxiv.org/html/2509.17765v1)
- [Qwen3-Omni GitHub](https://github.com/QwenLM/Qwen3-Omni)
- [Qwen2.5-Omni GitHub](https://github.com/QwenLM/Qwen2.5-Omni)
- [ERVQ: Enhanced RVQ](https://arxiv.org/html/2410.12359)
- [RVQ Explainer](https://drscotthawley.github.io/blog/posts/2023-06-12-RVQ.html)
- [Neural Audio Codecs](https://kyutai.org/codec-explainer)
- [BitTTS Quantization](https://arxiv.org/html/2506.03515v1)
- [EnCodec AudioCraft](https://audiocraft.metademolab.com/encodec.html)
- [llama.cpp Quantization](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
- [vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
