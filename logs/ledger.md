## 2026-01-28 20:15
Done: Text completion example for Qwen3-Omni BF16
- Created candle-examples/examples/qwen3_omni_text/main.rs
- Fixed config parsing: thinker_config.text_config extraction
- Fixed tensor prefix: added "thinker" to VarBuilder
- Made audio_embed and talker_head optional for text-only models
- Model loads config correctly (hidden_size=2048, 48 layers, vocab=152064)

Issue: OOM on CPU mode (66GB BF16 → 132GB F32)
- Lyuda has Vulkan/wgpu GPU, not CUDA
- Need wgpu backend integration for GPU inference

Next: Integrate candle-wgpu or find workaround for BF16 on CPU
