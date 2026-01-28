# Qwen3-Omni Text Completion - Quick Start

Минимальная инструкция для запуска text-only completion на Lyuda.

## 1. Подключение к Lyuda

```bash
sshpass -p '1q2w3e' ssh -p 2233 lluda@127.0.0.1
```

## 2. Установка зависимостей (один раз)

```bash
echo "1q2w3e" | sudo -S apt-get install -y libssl-dev pkg-config
```

## 3. Получение последних изменений

```bash
cd ~/candle-16b
git pull origin qwen3-omni-16b
```

## 4. Компиляция

```bash
cargo build --release --example qwen3_omni_text
```

Время: ~5-10 минут (первая компиляция), ~30 сек (инкрементальная).

## 5. Найти путь к весам

```bash
# Найти snapshot ID
SNAPSHOT_DIR=$(ls -t /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/ | head -1)
WEIGHT_PATH="/home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/snapshots/$SNAPSHOT_DIR"

echo "Weights path: $WEIGHT_PATH"

# Проверить наличие файлов
ls "$WEIGHT_PATH"/*.safetensors | wc -l
# Должно показать 15 (или 1, если single file)
```

## 6. Запуск

### Простой тест

```bash
./target/release/examples/qwen3_omni_text \
  --weight-path "$WEIGHT_PATH" \
  --prompt "Какая столица России?" \
  --sample-len 50
```

### С параметрами

```bash
./target/release/examples/qwen3_omni_text \
  --weight-path "$WEIGHT_PATH" \
  --prompt "What is machine learning?" \
  --sample-len 100 \
  --temperature 0.7 \
  --top-p 0.9
```

## Ожидаемый результат

```
Loading tokenizer from: .../tokenizer.json
Loading config from: .../config.json
Device: Cuda(0), dtype: BF16
Loading 15 shard files
Loaded the model in 12.5s
Prompt tokens: 8 tokens
Какая столица России?Москва
50 tokens generated (4.2 token/s)
```

## Troubleshooting

### Ошибка: "openssl-sys"
```bash
echo "1q2w3e" | sudo -S apt-get install -y libssl-dev pkg-config
cargo clean
cargo build --release --example qwen3_omni_text
```

### Ошибка: "Cannot find tokenizer.json"
Проверьте путь к весам:
```bash
ls "$WEIGHT_PATH"/tokenizer.json
```

### Out of memory
- Модель: ~66GB BF16
- Strix Halo (128GB): должно работать
- Проверьте: `rocm-smi` или `free -h`

## Полная документация

- Example README: `candle-examples/examples/qwen3_omni_text/README.md`
- Setup guide: `docs/QWEN3_OMNI_TEXT_SETUP.md`
- Changes log: `docs/CHANGES.md`

## Следующие шаги

После успешного запуска:
1. Тест на разных промптах (русский, английский)
2. Измерение производительности (tokens/s, latency)
3. Тест на длинных контекстах (1K, 8K, 16K)
4. Интеграция audio encoder для полной модели
