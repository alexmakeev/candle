# Qwen3-Omni Text-Only Completion Example

Минимальный example для тестирования Thinker модели (30B MoE с 3B активными параметрами) из Qwen3-Omni без audio encoder.

## Архитектура

Этот example использует только компонент **Thinker** из полной Qwen3-Omni архитектуры:

```
Text Input → Thinker (30B MoE) → Text Output
```

Полная архитектура Qwen3-Omni (не используется здесь):
```
Audio → AuT Encoder → Thinker → Talker → Code2Wav → Audio
```

## Особенности Thinker

- **Размер**: 30B параметров MoE, 3B активных на токен
- **Архитектура**: Qwen3-подобная с MoE каждый 2-й слой
- **GQA**: 32 heads, 8 KV heads
- **Context**: до 32K tokens
- **Vocab**: 151936 tokens

## Компиляция

```bash
cd /home/alexii/lluda/candle-16b
cargo build --release --example qwen3_omni_text
```

## Использование

### На удаленной машине (Lyuda)

```bash
# SSH подключение
sshpass -p '1q2w3e' ssh -p 2233 lluda@127.0.0.1

# Запуск
cd ~/candle-16b
./target/release/examples/qwen3_omni_text \
  --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/SNAPSHOT_ID \
  --prompt "Какая столица России?" \
  --sample-len 50
```

### Локально (для тестирования)

```bash
./target/release/examples/qwen3_omni_text \
  --weight-path /path/to/qwen3-omni-weights \
  --prompt "What is the capital of Russia?" \
  --sample-len 50 \
  --temperature 0.7 \
  --top-p 0.9
```

## Параметры

- `--weight-path` (обязательный): путь к директории с весами
- `--tokenizer-file` (опциональный): путь к tokenizer.json (по умолчанию ищет в weight-path)
- `--prompt` (обязательный): текстовый prompt
- `--sample-len`: максимальная длина генерации в токенах (по умолчанию 100)
- `--temperature`: температура сэмплирования (опциональный)
- `--top-p`: nucleus sampling cutoff (опциональный)
- `--repeat-penalty`: штраф за повторы (по умолчанию 1.1)
- `--repeat-last-n`: размер окна для repeat penalty (по умолчанию 64)
- `--seed`: random seed (по умолчанию 299792458)
- `--cpu`: использовать CPU вместо GPU
- `--tracing`: включить tracing (создает trace-timestamp.json)

## Структура весов

Example поддерживает три формата:

1. **Sharded с индексом**:
   ```
   model.safetensors.index.json
   model-00001-of-00015.safetensors
   model-00002-of-00015.safetensors
   ...
   ```

2. **Единый файл**:
   ```
   model.safetensors
   ```

3. **Множественные шарды без индекса**:
   ```
   model-00001-of-00015.safetensors
   model-00002-of-00015.safetensors
   ...
   ```

## Ожидаемый вывод

```
avx: true, neon: false, simd128: false, f16c: true
temp: 0.70 repeat-penalty: 1.10 repeat-last-n: 64
Loading tokenizer from: /path/to/tokenizer.json
Loading config from: /path/to/config.json
Device: Cuda(0), dtype: BF16
Loading 15 shard files
Loaded the model in 12.5s
Prompt tokens: 8 tokens
Какая столица России?Москва
50 tokens generated (4.2 token/s)
```

## Технические детали

### KV Cache
KV cache активирован и управляется автоматически внутри Thinker. На первом forward pass кэш пустой, затем накапливается.

### MoE Layers
MoE активируется каждый 2-й слой (decoder_sparse_step=2):
- Layers 1, 3, 5, ... : Dense MLP
- Layers 2, 4, 6, ... : Sparse MoE (64 experts, 4 active)

### Memory Usage
Для BF16 на GPU (~66 GB):
- Model weights: ~66 GB
- KV cache: зависит от длины контекста
- Activations: ~1-2 GB

## Troubleshooting

### OOM (Out of Memory)
- Используйте меньший batch size (по умолчанию 1)
- Уменьшите sample_len
- Используйте CPU: `--cpu`

### Неправильная загрузка весов
Убедитесь что:
- Путь содержит все safetensors файлы
- tokenizer.json и config.json в той же директории
- Права доступа на чтение

### Некорректный вывод
- Проверьте что используется правильный tokenizer
- Попробуйте другие параметры temperature/top_p
- Убедитесь что веса загружены полностью

## См. также

- Полная документация: `/home/alexii/lluda/candle-16b/docs/CHANGES.md`
- Модуль модели: `candle-transformers/src/models/qwen3_omni/`
- Оригинальный репозиторий: https://huggingface.co/Qwen/Qwen3-Omni
