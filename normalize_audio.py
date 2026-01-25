#!/usr/bin/env python3
"""Normalize audio to target RMS"""
import sys
import numpy as np
from scipy.io import wavfile

input_wav = sys.argv[1]
output_wav = sys.argv[2]
target_rms = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

sr, audio = wavfile.read(input_wav)
if len(audio.shape) > 1:
    audio = audio[:, 0]

audio_float = audio.astype(np.float32) / 32768.0

# Current RMS
current_rms = np.sqrt(np.mean(audio_float**2))
print(f"Current RMS: {current_rms:.4f}")

# Scale to target
scale = target_rms / (current_rms + 1e-8)
normalized = audio_float * scale

# Clip
normalized = np.clip(normalized, -1.0, 1.0)

# Convert back
audio_int16 = (normalized * 32767).astype(np.int16)

wavfile.write(output_wav, sr, audio_int16)
print(f"Saved: {output_wav}")
print(f"New RMS: {target_rms:.4f}")
