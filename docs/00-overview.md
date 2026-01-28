# Qwen3-Omni Architecture Overview (BF16 Version)

## Model Components

```
Audio Input (16kHz)
    │
    ▼
┌─────────────────┐
│  AuT Encoder    │  650M params - Custom audio encoder (NOT Whisper)
│  (Audio→Token)  │  Encodes 16kHz audio into discrete tokens
└────────┬────────┘
         │ audio_tokens [batch, seq] ∈ [0, 4095]
         ▼
┌─────────────────┐
│    Thinker      │  30B MoE (3B active) - Main reasoning model
│  (Think+Speak)  │  Generates text + talker tokens
└────────┬────────┘
         │ talker_tokens [batch, seq]
         ▼
┌─────────────────┐
│    Talker       │  3B MoE (0.3B active) - Speech synthesis
│  (Token→Codec)  │  Generates multi-codebook audio tokens
└────────┬────────┘
         │ codec_tokens [batch, seq, 4 codebooks]
         ▼
┌─────────────────┐
│   Code2Wav      │  ~200M params - Neural vocoder
│  (Codec→Audio)  │  Converts codebook tokens to waveform
└─────────────────┘
         │
         ▼
    Audio Output (16kHz)
```

## Model Sizes Summary

| Component | Total Params | Active Params | Note |
|-----------|-------------|---------------|------|
| AuT Encoder | 650M | 650M | Dense encoder |
| Thinker | 30B | 3B | MoE: 64 experts, 4 active |
| Talker | 3B | 0.3B | MoE: 32 experts, 4 active |
| Code2Wav | 200M | 200M | Dense vocoder |
| **TOTAL** | ~33.85B | ~4.15B | BF16 = ~68 GB |

## Key Files

| File | Purpose |
|------|---------|
| `mod.rs` | Main Qwen3Omni struct, integration |
| `config.rs` | All configuration structures |
| `thinker.rs` | 30B MoE reasoning model |
| `talker.rs` | 3B MoE speech synthesis |
| `code2wav.rs` | Neural vocoder |
| `aut_encoder.rs` | Audio-to-Token encoder |
| `audio.rs` | Mel spectrogram, WAV I/O |

## Documentation Files

- [01-module-structure.md](01-module-structure.md) - Module organization
- [02-thinker.md](02-thinker.md) - Thinker (30B MoE) details
- [03-talker.md](03-talker.md) - Talker (3B MoE TTS) details
- [04-code2wav.md](04-code2wav.md) - Neural vocoder details
- [05-audio-input.md](05-audio-input.md) - Audio processing pipeline
- [06-configuration.md](06-configuration.md) - All configuration parameters
