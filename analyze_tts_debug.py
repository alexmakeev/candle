#!/usr/bin/env python3
"""
Qwen3-Omni TTS Debug Analyzer
Задачи:
1. Анализ спектрограмм (реальная речь vs генерация)
2. Проверка codec tokens
3. Изолированное тестирование Code2Wav
4. Поиск этапа где теряется информация
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import librosa.display
import json
import sys
from pathlib import Path

def analyze_audio_spectrogram(wav_path, title="Audio Spectrogram"):
    """Анализ спектрограммы аудио"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {wav_path}")
    print(f"{'='*60}")
    
    # Загрузка
    sr, audio = wavfile.read(wav_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # mono
    
    audio_float = audio.astype(np.float32) / 32768.0
    
    # Базовая статистика
    print(f"\nBasic Statistics:")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Duration: {len(audio) / sr:.2f} sec")
    print(f"  Samples: {len(audio)}")
    print(f"  Min value: {audio_float.min():.4f}")
    print(f"  Max value: {audio_float.max():.4f}")
    print(f"  Mean: {audio_float.mean():.6f}")
    print(f"  Std: {audio_float.std():.6f}")
    print(f"  RMS: {np.sqrt(np.mean(audio_float**2)):.6f}")
    
    # Проверка на шум vs речь
    # Речь имеет структурированную энергию, шум — равномерную
    energy = np.abs(audio_float)
    energy_variance = np.var(energy)
    print(f"  Energy variance: {energy_variance:.6f}")
    
    # Zero crossing rate (речь имеет переменный ZCR, шум — постоянный)
    zcr = librosa.zero_crossings(audio_float, pad=False).sum()
    zcr_rate = zcr / len(audio_float)
    print(f"  Zero crossing rate: {zcr_rate:.6f}")
    
    # Спектральный центроид (речь ~500-2000Hz, шум — широкий спектр)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_float, sr=sr)[0]
    print(f"  Spectral centroid mean: {spectral_centroids.mean():.1f} Hz")
    print(f"  Spectral centroid std: {spectral_centroids.std():.1f} Hz")
    
    # Спектрограмма
    D = librosa.stft(audio_float)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    # Анализ частотного распределения
    freq_bins = librosa.fft_frequencies(sr=sr)
    avg_spectrum = np.mean(np.abs(D), axis=1)
    
    # Энергия в диапазонах
    low_freq_energy = np.mean(avg_spectrum[freq_bins < 500])
    speech_freq_energy = np.mean(avg_spectrum[(freq_bins >= 500) & (freq_bins < 4000)])
    high_freq_energy = np.mean(avg_spectrum[freq_bins >= 4000])
    
    print(f"\nFrequency Energy Distribution:")
    print(f"  Low (<500 Hz): {low_freq_energy:.2f}")
    print(f"  Speech (500-4000 Hz): {speech_freq_energy:.2f}")
    print(f"  High (>4000 Hz): {high_freq_energy:.2f}")
    print(f"  Speech ratio: {speech_freq_energy / (low_freq_energy + speech_freq_energy + high_freq_energy):.2%}")
    
    # Вердикт
    print(f"\n{'='*60}")
    is_noise = False
    reasons = []
    
    if energy_variance < 0.001:
        is_noise = True
        reasons.append("Very low energy variance (uniform noise)")
    
    if spectral_centroids.mean() > 6000:
        is_noise = True
        reasons.append("High spectral centroid (white noise)")
    
    if speech_freq_energy / (low_freq_energy + speech_freq_energy + high_freq_energy) < 0.3:
        is_noise = True
        reasons.append("Low speech frequency energy ratio")
    
    if is_noise:
        print("VERDICT: This looks like NOISE ❌")
        for reason in reasons:
            print(f"  - {reason}")
    else:
        print("VERDICT: This looks like SPEECH ✓")
    print(f"{'='*60}\n")
    
    # Визуализация
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # Waveform
    time = np.arange(len(audio_float)) / sr
    axes[0].plot(time, audio_float)
    axes[0].set_title(f'{title} - Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Spectrogram
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1], cmap='viridis')
    axes[1].set_title(f'{title} - Spectrogram')
    axes[1].set_ylim(0, 8000)
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    # Spectrum
    axes[2].plot(freq_bins[:len(avg_spectrum)], avg_spectrum)
    axes[2].set_title(f'{title} - Average Spectrum')
    axes[2].set_xlabel('Frequency (Hz)')
    axes[2].set_ylabel('Magnitude')
    axes[2].set_xlim(0, 8000)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = wav_path.replace('.wav', '_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Spectrogram saved: {output_path}")
    plt.close()
    
    return {
        'is_noise': is_noise,
        'reasons': reasons,
        'stats': {
            'duration': len(audio) / sr,
            'rms': float(np.sqrt(np.mean(audio_float**2))),
            'energy_variance': float(energy_variance),
            'zcr': float(zcr_rate),
            'spectral_centroid_mean': float(spectral_centroids.mean()),
            'speech_energy_ratio': float(speech_freq_energy / (low_freq_energy + speech_freq_energy + high_freq_energy))
        }
    }

def analyze_codec_tokens(tokens_file):
    """Анализ codec tokens"""
    print(f"\n{'='*60}")
    print(f"Analyzing codec tokens: {tokens_file}")
    print(f"{'='*60}")
    
    with open(tokens_file, 'r') as f:
        tokens = json.load(f)
    
    print(f"\nToken Statistics:")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Unique tokens: {len(set(tokens))}")
    print(f"  Min token: {min(tokens)}")
    print(f"  Max token: {max(tokens)}")
    print(f"  Mean: {np.mean(tokens):.2f}")
    print(f"  Std: {np.std(tokens):.2f}")
    
    # Проверка диапазона (для Qwen3-Omni codec tokens должны быть в разумном диапазоне)
    if min(tokens) < 0 or max(tokens) > 10000:
        print(f"  WARNING: Token range suspicious! Expected 0-10000")
    
    # Проверка на константы (если все токены одинаковые = проблема)
    if len(set(tokens)) < 10:
        print(f"  WARNING: Very few unique tokens ({len(set(tokens))})! Likely a bug.")
    
    # Распределение токенов
    hist, bins = np.histogram(tokens, bins=50)
    plt.figure(figsize=(12, 6))
    plt.bar(bins[:-1], hist, width=(bins[1]-bins[0])*0.9)
    plt.title('Codec Token Distribution')
    plt.xlabel('Token Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    output_path = tokens_file.replace('.json', '_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved: {output_path}")
    plt.close()

def compare_audios(real_wav, generated_wav):
    """Сравнение реальной речи и генерации"""
    print(f"\n{'='*60}")
    print(f"COMPARISON: Real Speech vs Generated")
    print(f"{'='*60}")
    
    real_result = analyze_audio_spectrogram(real_wav, "Real Speech")
    gen_result = analyze_audio_spectrogram(generated_wav, "Generated TTS")
    
    print(f"\n{'='*60}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nReal Speech:")
    print(f"  Verdict: {'NOISE ❌' if real_result['is_noise'] else 'SPEECH ✓'}")
    print(f"  RMS: {real_result['stats']['rms']:.6f}")
    print(f"  Speech energy ratio: {real_result['stats']['speech_energy_ratio']:.2%}")
    
    print(f"\nGenerated TTS:")
    print(f"  Verdict: {'NOISE ❌' if gen_result['is_noise'] else 'SPEECH ✓'}")
    print(f"  RMS: {gen_result['stats']['rms']:.6f}")
    print(f"  Speech energy ratio: {gen_result['stats']['speech_energy_ratio']:.2%}")
    
    print(f"\nDifferences:")
    print(f"  RMS ratio: {gen_result['stats']['rms'] / real_result['stats']['rms']:.2f}x")
    print(f"  Speech energy ratio diff: {(gen_result['stats']['speech_energy_ratio'] - real_result['stats']['speech_energy_ratio']) * 100:.1f}%")
    
    if gen_result['is_noise'] and not real_result['is_noise']:
        print(f"\n⚠️  PROBLEM CONFIRMED: Generated output is noise while real speech is valid!")
        print(f"   Root cause likely in: Talker → Code2Wav pipeline")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Analyze single audio: python analyze_tts_debug.py <wav_file>")
        print("  Compare two audios: python analyze_tts_debug.py <real_wav> <generated_wav>")
        print("  Analyze tokens: python analyze_tts_debug.py --tokens <tokens.json>")
        sys.exit(1)
    
    if sys.argv[1] == "--tokens":
        analyze_codec_tokens(sys.argv[2])
    elif len(sys.argv) == 2:
        analyze_audio_spectrogram(sys.argv[1])
    else:
        compare_audios(sys.argv[1], sys.argv[2])
