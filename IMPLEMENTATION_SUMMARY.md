# Qwen3-Omni Text Completion - Implementation Summary

Дата: 2026-01-28
Статус: Реализован минимальный text-only completion, требуется тестирование

## Что реализовано

### 1. Example: qwen3_omni_text

**Файл**: `candle-examples/examples/qwen3_omni_text/main.rs`

Минимальный пример для text-only inference через Thinker модель:

- Загрузка tokenizer через `tokenizers` crate
- Загрузка config из JSON (ThinkerConfig)
- Загрузка SafeTensors весов через VarBuilder
- Автоматическое определение формата весов:
  - Sharded с index.json
  - Единый model.safetensors
  - Множественные model-*.safetensors без индекса
- Autoregressive генерация:
  - Первый forward pass с полным промптом
  - Затем токен-за-токеном с KV cache
- Параметры сэмплирования:
  - Temperature
  - Top-p (nucleus sampling)
  - Repeat penalty
- Детектирование EOS tokens (<|endoftext|>, <|im_end|>)
- Streaming вывод через TokenOutputStream

**Размер**: ~330 строк кода

### 2. Документация

**Файлы**:
- `candle-examples/examples/qwen3_omni_text/README.md` - описание example
- `docs/QWEN3_OMNI_TEXT_SETUP.md` - полный setup guide
- `docs/CHANGES.md` - лог изменений
- `QWEN3_OMNI_QUICKSTART.md` - краткая инструкция для быстрого старта

**Содержание**:
- Инструкции по компиляции
- Параметры запуска
- Ожидаемый вывод
- Troubleshooting (OpenSSL, OOM, неправильные пути)
- Memory usage breakdown
- Performance benchmarks (ориентировочные)

## Архитектура

### Используется только Thinker

```
Text Tokens → Thinker (30B MoE, 3B active) → Text Logits → Sampling → Output
                                             ↘ Talker Tokens (игнорируются)
```

### Полная архитектура Qwen3-Omni (не используется в этом example)

```
Audio → AuT Encoder → Thinker → Talker → Code2Wav → Audio
                         ↓
                    Text Output
```

### Особенности Thinker

- **Размер**: 30B параметров MoE
- **Активные**: 3B параметров на токен
- **Архитектура**: 40 layers, GQA (32 heads, 8 KV heads)
- **MoE**: 64 experts, 4 active per token, каждый 2-й слой
- **Hidden size**: 4096
- **Vocab**: 151936 tokens
- **RoPE**: theta = 1M
- **Max context**: 32K tokens

## Структура проекта

```
candle-16b/
├── candle-transformers/src/models/qwen3_omni/
│   ├── mod.rs           # Экспорт модулей
│   ├── config.rs        # Config, ThinkerConfig, etc.
│   ├── thinker.rs       # Thinker модель (используется)
│   ├── aut_encoder.rs   # AuT encoder (не используется)
│   ├── talker.rs        # Talker модель (не используется)
│   ├── code2wav.rs      # Vocoder (не используется)
│   └── audio.rs         # Audio utils (не используется)
├── candle-examples/examples/qwen3_omni_text/
│   ├── main.rs          # Example код
│   └── README.md        # Документация example
├── docs/
│   ├── CHANGES.md                  # Лог изменений
│   └── QWEN3_OMNI_TEXT_SETUP.md   # Setup guide
├── QWEN3_OMNI_QUICKSTART.md       # Быстрый старт
└── IMPLEMENTATION_SUMMARY.md      # Этот файл
```

## Зависимости

### Системные (для компиляции)

```bash
sudo apt-get install -y libssl-dev pkg-config
```

Причина: `tokenizers` crate зависит от `openssl-sys`

### Rust crates

Используются существующие dependencies проекта:
- `candle` - ML framework
- `candle_nn` - Neural network layers
- `candle_transformers` - Transformer models
- `candle_examples` - Utility functions
- `tokenizers` - HuggingFace tokenizers
- `clap` - CLI argument parsing
- `anyhow` - Error handling
- `serde_json` - Config parsing

Никаких новых зависимостей не требуется.

## Компиляция и запуск

### Быстрый старт (на Lyuda)

```bash
# 1. Подключение
sshpass -p '1q2w3e' ssh -p 2233 lluda@127.0.0.1

# 2. Установка зависимостей (один раз)
echo "1q2w3e" | sudo -S apt-get install -y libssl-dev pkg-config

# 3. Обновление кода
cd ~/candle-16b
git pull origin qwen3-omni-16b

# 4. Компиляция
cargo build --release --example qwen3_omni_text

# 5. Запуск
SNAPSHOT_DIR=$(ls -t /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/ | head -1)
./target/release/examples/qwen3_omni_text \
  --weight-path "/home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/$SNAPSHOT_DIR" \
  --prompt "Какая столица России?" \
  --sample-len 50
```

См. `QWEN3_OMNI_QUICKSTART.md` для подробностей.

## Тестирование

### Статус

**Не протестировано** - требуется запуск на Lyuda с реальными весами.

### План тестирования

1. **Компиляция на Lyuda**
   - Установить libssl-dev
   - Скомпилировать в release mode
   - Проверить размер бинарника

2. **Загрузка модели**
   - Найти snapshot directory
   - Проверить наличие всех 15 safetensors файлов
   - Загрузить модель (ожидается 10-15 сек)
   - Проверить Device: Cuda(0), dtype: BF16

3. **Простая генерация**
   - Prompt: "Hello"
   - Sample len: 5 tokens
   - Проверить что генерирует без ошибок

4. **Реальная генерация**
   - Русский: "Какая столица России?"
   - Английский: "What is machine learning?"
   - Ожидается осмысленный ответ

5. **Производительность**
   - Измерить tokens/s
   - Ожидается: 3-5 tokens/s на Strix Halo
   - Измерить first token latency

6. **Длинные контексты**
   - Тест на 1K tokens
   - Тест на 8K tokens
   - Тест на 16K tokens (может быть OOM)
   - Измерить memory usage (rocm-smi)

### Ожидаемые метрики (Strix Halo, 128GB)

| Metric | Expected Value |
|--------|---------------|
| Model load time | 10-15 seconds |
| First token latency | 1-2 seconds |
| Token throughput | 3-5 tokens/s |
| Memory usage (1K ctx) | ~70 GB |
| Memory usage (8K ctx) | ~80 GB |
| Memory usage (16K ctx) | ~90 GB |
| Max context (safe) | 16K tokens |

## Known Issues

### 1. OpenSSL dependency

**Проблема**: Компиляция требует libssl-dev на системе

**Причина**: tokenizers → native-tls → openssl-sys

**Решение**: Установить перед компиляцией:
```bash
echo "1q2w3e" | sudo -S apt-get install -y libssl-dev pkg-config
```

**Альтернатива**: Использовать vendored OpenSSL (требует изменений в Cargo.toml)

### 2. Config parsing

**Потенциальная проблема**: config.json может не точно соответствовать ThinkerConfig

**Решение**: ThinkerConfig имеет Default implementation со всеми значениями по умолчанию

### 3. KV Cache не очищается между запросами

**Проблема**: В текущей реализации KV cache накапливается внутри модели

**Решение**: Для multiple prompts нужно добавить `model.clear_kv_cache()` между запросами

**Статус**: Не критично для single-shot inference

## Следующие шаги

### Immediate (после успешного тестирования)

1. **Тестирование на Lyuda**
   - Скомпилировать
   - Запустить на реальных весах
   - Проверить корректность вывода
   - Измерить производительность

2. **Отладка (если нужно)**
   - Исправить ошибки загрузки
   - Настроить параметры генерации
   - Оптимизировать memory usage

3. **Документация результатов**
   - Обновить CHANGES.md с результатами тестов
   - Добавить реальные benchmarks в README
   - Создать примеры успешных запусков

### Short-term (следующие 1-2 недели)

1. **Улучшение example**
   - Добавить batch inference
   - Интерактивный режим (chat)
   - Сохранение/загрузка KV cache

2. **Интеграция audio encoder**
   - Example для audio → text
   - Загрузка AuT encoder весов
   - Тестирование на audio файлах

3. **Полная модель**
   - Интеграция Talker
   - Интеграция Code2Wav
   - End-to-end speech-to-speech example

### Long-term (месяц+)

1. **Квантизация**
   - Q8_0 quantization для Thinker
   - Уменьшение memory footprint до ~33GB
   - DP4a оптимизация для AMD

2. **Оптимизация**
   - Flash Attention integration
   - Kernel fusion
   - Batch inference optimization

3. **Production deployment**
   - Server mode
   - REST API
   - WebSocket streaming

## Вопросы для тестирования

При первом запуске проверить:

1. **Загрузка весов**
   - Все ли 15 safetensors файлов загружены?
   - Правильный ли dtype (BF16)?
   - Какое время загрузки?

2. **Генерация**
   - Корректный ли вывод для простых промптов?
   - Работает ли на русском языке?
   - Детектируется ли EOS токен правильно?

3. **Производительность**
   - Сколько tokens/s?
   - Первый токен latency?
   - Memory usage (rocm-smi)?

4. **Проблемы**
   - Есть ли ошибки/warnings?
   - OOM на каком контексте?
   - Нужны ли дополнительные флаги/параметры?

## Контакты и ресурсы

- **Проект**: /home/alexii/lluda/candle-16b (ветка qwen3-omni-16b)
- **Remote**: Lyuda (SSH port 2233)
- **Веса**: /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/
- **Документация**: docs/ директория
- **HuggingFace**: https://huggingface.co/Qwen/Qwen3-Omni

## Changelog

- 2026-01-28 13:30 - Создан example qwen3_omni_text/main.rs
- 2026-01-28 13:45 - Добавлена документация (README.md)
- 2026-01-28 13:50 - Создан setup guide (QWEN3_OMNI_TEXT_SETUP.md)
- 2026-01-28 13:55 - Создан quickstart (QWEN3_OMNI_QUICKSTART.md)
- 2026-01-28 14:00 - Создан implementation summary (этот файл)

См. `docs/CHANGES.md` для подробного лога изменений.
