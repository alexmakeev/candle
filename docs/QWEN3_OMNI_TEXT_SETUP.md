# Qwen3-Omni Text Completion Setup

## Prerequisites

### System Dependencies

На машине для компиляции требуется установить:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y libssl-dev pkg-config

# или для запасного варианта
sudo apt-get install -y libssl-dev pkg-config build-essential
```

### Rust

Убедитесь, что установлен Rust toolchain:

```bash
rustc --version  # должно быть >= 1.70
```

## Компиляция

### На локальной машине (jam)

Если libssl-dev недоступен, используйте удаленную компиляцию на Люде.

### На удаленной машине (Lyuda)

```bash
# Подключение через SSH
sshpass -p '1q2w3e' ssh -p 2233 lluda@127.0.0.1

# Установить зависимости (если еще не установлены)
echo "1q2w3e" | sudo -S apt-get install -y libssl-dev pkg-config

# Перейти в проект
cd ~/candle-16b

# Обновить из git (если изменения были сделаны локально и запушены)
git pull origin qwen3-omni-16b

# Компиляция в release mode
cargo build --release --example qwen3_omni_text
```

Время компиляции: ~5-10 минут (первая сборка), ~30 сек (инкрементальная).

### Проверка компиляции

```bash
ls -lh target/release/examples/qwen3_omni_text
# Должен показать бинарный файл размером ~50-100 MB
```

## Запуск

### Найти snapshot ID модели

```bash
ls /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/
# Выберите нужный snapshot (обычно последний)
```

### Базовый запуск

```bash
./target/release/examples/qwen3_omni_text \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/SNAPSHOT_ID \
  --prompt "Какая столица России?" \
  --sample-len 50
```

### С параметрами генерации

```bash
./target/release/examples/qwen3_omni_text \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/SNAPSHOT_ID \
  --prompt "What is the capital of Russia?" \
  --sample-len 100 \
  --temperature 0.7 \
  --top-p 0.9 \
  --repeat-penalty 1.1 \
  --repeat-last-n 64
```

### На CPU (для отладки)

```bash
./target/release/examples/qwen3_omni_text \
  --weight-path /path/to/weights \
  --prompt "Test prompt" \
  --cpu
```

## Ожидаемый вывод

Успешный запуск должен показать:

```
avx: true, neon: false, simd128: false, f16c: true
temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
Loading tokenizer from: /home/lluda/.cache/.../tokenizer.json
Loading config from: /home/lluda/.cache/.../config.json
Config: ThinkerConfig { hidden_size: 4096, num_hidden_layers: 40, ... }
Retrieved files in 0.01s
Device: Cuda(0), dtype: BF16
Loading 15 shard files
  [1] /home/lluda/.cache/.../model-00001-of-00015.safetensors
  [2] /home/lluda/.cache/.../model-00002-of-00015.safetensors
  ...
  [15] /home/lluda/.cache/.../model-00015-of-00015.safetensors
Loading Thinker model...
Loaded the model in 12.5s
Prompt tokens: 8 tokens
Какая столица России?Москва
50 tokens generated (4.2 token/s)
```

## Troubleshooting

### Ошибка компиляции: "openssl-sys"

**Проблема**: Не установлен libssl-dev

**Решение**:
```bash
echo "1q2w3e" | sudo -S apt-get install -y libssl-dev pkg-config
```

### Ошибка запуска: "Cannot find tokenizer.json"

**Проблема**: Неправильный путь к весам

**Решение**:
- Проверьте что путь содержит tokenizer.json
- Используйте полный путь к snapshot директории
- Или укажите явно: `--tokenizer-file /path/to/tokenizer.json`

### Ошибка запуска: "Out of memory"

**Проблема**: Недостаточно VRAM (модель ~66GB BF16)

**Решение**:
- Убедитесь что используется GPU с 64GB+ памяти
- На Strix Halo (128GB unified): должно работать
- Альтернатива: используйте `--cpu` (очень медленно)

### Медленная генерация (< 1 token/s на GPU)

**Возможные причины**:
- Используется CPU вместо GPU (проверьте вывод "Device:")
- Swap на диск из-за нехватки RAM
- Проблемы с CUDA/ROCm драйверами

**Диагностика**:
```bash
# Проверить использование GPU
nvidia-smi  # для NVIDIA
rocm-smi    # для AMD
```

### Ошибка "config.json parse error"

**Проблема**: config.json не соответствует ThinkerConfig

**Решение**:
- Убедитесь что используете Qwen3-Omni, а не Qwen2/Qwen3
- Проверьте что config.json содержит поля для Thinker
- См. документацию в `candle-transformers/src/models/qwen3_omni/config.rs`

## Тестовые команды

### Простой тест (должен работать мгновенно после загрузки)

```bash
./target/release/examples/qwen3_omni_text \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/SNAPSHOT \
  --prompt "Hello" \
  --sample-len 5
```

### Тест генерации

```bash
./target/release/examples/qwen3_omni_text \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/SNAPSHOT \
  --prompt "Write a haiku about programming:" \
  --sample-len 50 \
  --temperature 0.8
```

## Memory Usage

Ориентировочное использование памяти для BF16 на GPU:

| Компонент | Размер |
|-----------|--------|
| Model weights | ~66 GB |
| KV cache (1K ctx) | ~2 GB |
| KV cache (8K ctx) | ~16 GB |
| KV cache (32K ctx) | ~64 GB |
| Activations | ~1-2 GB |
| **Total (1K ctx)** | **~70 GB** |
| **Total (32K ctx)** | **~133 GB** |

На Strix Halo с 128GB unified memory: должно работать до ~16K контекста комфортно.

## Performance Benchmarks

На AMD Ryzen AI Max+ 395 (Radeon 8060S, 64GB VRAM):

| Metric | Expected Value |
|--------|---------------|
| Load time | 10-15 seconds |
| First token latency | 1-2 seconds |
| Token throughput | 3-5 tokens/s |
| Memory usage (1K ctx) | 70 GB |

Эти числа ориентировочные и могут отличаться в зависимости от конфигурации.

## Next Steps

После успешного запуска text completion:

1. Интеграция audio encoder (AuT)
2. Добавление Talker для speech synthesis
3. Полная end-to-end speech-to-speech pipeline
4. Квантизация (Q8_0) для меньшего использования памяти

См. также: `/home/alexii/lluda/candle-16b/docs/CHANGES.md`
