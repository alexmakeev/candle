# Candle wgpu Backend for AMD Strix Halo

## Remote Machine: Lyuda

### SSH Connection

**ВАЖНО: Всегда использовать 127.0.0.1, не внешний IP!**

**Основной туннель (порт 2233)** — autossh сервис на Люде, переподнимается автоматически:
```bash
sshpass -p '1q2w3e' ssh -p 2233 lluda@127.0.0.1
```

**Запасной туннель (порт 2222)** — ручной, запускается с Mac пользователя когда автотуннель недоступен:
```bash
# Команда на Mac для поднятия запасного туннеля:
# ssh -R 2222:192.168.5.126:22 alexii@jam.robobobr.ru -N

# Подключение через запасной:
sshpass -p '1q2w3e' ssh -p 2222 lluda@127.0.0.1
```

- **Host**: 127.0.0.1 (ТОЛЬКО через туннель, не внешний IP!)
- **Ports**: 2233 (auto, primary), 2222 (manual, backup)
- **User**: lluda
- **Password**: 1q2w3e
- **Sudo**: есть, тот же пароль (`echo "1q2w3e" | sudo -S ...`)

**Autossh сервис на Люде:**
```bash
# Статус
systemctl status autossh-tunnel
# Рестарт
echo "1q2w3e" | sudo -S systemctl restart autossh-tunnel
```

### Hardware
- **APU**: AMD Ryzen AI Max+ 395 (Strix Halo)
- **GPU**: Radeon 8060S (RDNA 3.5, gfx1151)
- **RAM**: 62 GB system / 64 GB VRAM (UMA Auto)
- **Total**: 128 GB unified memory

### System
- **OS**: Ubuntu 24.04.3 LTS
- **Kernel**: 6.14.0-36-generic
- **GPU Driver**: amdgpu (inbox)
- **Mesa**: 25.2.1 (kisak PPA)
- **Vulkan**: 1.3.275

### Model Location
```
/home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni/
```
- 15 safetensor files, ~66 GB total (BF16)

### Audio Devices
- **HDMI**: SMART TV (working)
- **Bluetooth**: CF22 speaker (working for output)

## Project Structure

**Локально (jam.robobobr.ru):**
```
/home/alexii/lluda/
├── candle-16b/   ← ветка qwen3-omni-16b (BF16, 16-бит)
└── candle-8b/    ← ветка main (Q8_0, 8-бит)
```

**На Люде:**
```
/home/lluda/
├── candle-16b/   ← ветка qwen3-omni-16b (BF16)
├── candle-8b/    ← ветка main (Q8_0)
└── old/          ← архив старых папок
```

**Workflow:**
```bash
# Локально: правки → commit → push
cd /home/alexii/lluda/candle-16b  # или candle-8b
git add . && git commit -m "..." && git push origin

# На Люде: pull → build
sshpass -p '1q2w3e' ssh -p 2233 lluda@127.0.0.1
cd ~/candle-16b && git pull origin && cargo build --release
```

**Цель:** 16-битная и 8-битная версии должны работать одинаково для сравнения.

## Project Goals

1. wgpu/Vulkan backend for Candle ML framework
2. Q8_0 quantization with DP4a optimization
3. Run Qwen3-Omni on Strix Halo GPU

## Key Files

- `candle-wgpu/` — wgpu backend implementation
- `candle-wgpu/src/quantized.rs` — Q8_0 matmul with DP4a
- `candle-transformers/src/models/qwen3_omni/` — Qwen3-Omni model
- `tensor-tools/src/streaming_quantize.rs` — streaming quantizer for large models

## Current Task

See `bd list` for Beads task tracker.
Epic: candle-dlm — Qwen3-Omni Q8_0 quantization and inference.
