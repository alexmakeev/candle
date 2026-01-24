//! Audio processing utilities for Qwen3-Omni
//!
//! Handles audio I/O, mel spectrogram computation, and preprocessing.

use candle::{Device, Result, Tensor};

/// Audio processor for preparing input audio
pub struct AudioProcessor {
    sample_rate: usize,
    n_fft: usize,
    hop_length: usize,
    n_mels: usize,
    device: Device,
}

impl AudioProcessor {
    pub fn new(
        sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
        device: &Device,
    ) -> Self {
        Self {
            sample_rate,
            n_fft,
            hop_length,
            n_mels,
            device: device.clone(),
        }
    }

    /// Convert raw PCM audio to mel spectrogram
    ///
    /// # Arguments
    /// * `pcm` - Raw audio samples at target sample rate
    ///
    /// # Returns
    /// * Mel spectrogram tensor [batch, n_mels, frames]
    pub fn pcm_to_mel(&self, pcm: &Tensor) -> Result<Tensor> {
        // For now, use CPU FFT from whisper module
        // TODO: Implement GPU-accelerated mel spectrogram via wgpu
        let pcm_data: Vec<f32> = pcm.to_vec1()?;

        // Generate mel filterbank (should be cached)
        let mel_filters = self.mel_filterbank();

        // Compute mel spectrogram
        let mel = crate::models::whisper::audio::log_mel_spectrogram_(
            &pcm_data,
            &mel_filters,
            self.n_fft,
            self.hop_length,
            self.n_mels,
            false,
        );

        let n_frames = mel.len() / self.n_mels;
        Tensor::from_slice(&mel, (1, self.n_mels, n_frames), &self.device)
    }

    /// Generate mel filterbank weights
    fn mel_filterbank(&self) -> Vec<f32> {
        let n_fft_bins = 1 + self.n_fft / 2;
        let mut filters = vec![0.0f32; self.n_mels * n_fft_bins];

        let f_min = 0.0;
        let f_max = self.sample_rate as f32 / 2.0;

        // Convert Hz to mel scale
        let hz_to_mel = |hz: f32| 2595.0 * (1.0 + hz / 700.0).log10();
        let mel_to_hz = |mel: f32| 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0);

        let mel_min = hz_to_mel(f_min);
        let mel_max = hz_to_mel(f_max);

        // Create mel points
        let mel_points: Vec<f32> = (0..=self.n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (self.n_mels + 1) as f32)
            .collect();

        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // Convert to FFT bin indices
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&hz| ((self.n_fft + 1) as f32 * hz / self.sample_rate as f32) as usize)
            .collect();

        // Create triangular filters
        for m in 0..self.n_mels {
            for k in bin_points[m]..bin_points[m + 1] {
                if k < n_fft_bins {
                    let denom = (bin_points[m + 1] - bin_points[m]) as f32;
                    if denom > 0.0 {
                        filters[m * n_fft_bins + k] = (k - bin_points[m]) as f32 / denom;
                    }
                }
            }
            for k in bin_points[m + 1]..bin_points[m + 2] {
                if k < n_fft_bins {
                    let denom = (bin_points[m + 2] - bin_points[m + 1]) as f32;
                    if denom > 0.0 {
                        filters[m * n_fft_bins + k] = (bin_points[m + 2] - k) as f32 / denom;
                    }
                }
            }
        }

        filters
    }
}

/// Load audio from file or raw bytes
///
/// Supports WAV format at 16kHz mono
pub fn load_audio(data: &[u8], target_sample_rate: usize) -> Result<Vec<f32>> {
    // Simple WAV parser for 16-bit PCM
    // For production, use hound or symphonia crate

    if data.len() < 44 {
        return Err(candle::Error::Msg("WAV file too short".into()));
    }

    // Check RIFF header
    if &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(candle::Error::Msg("Not a valid WAV file".into()));
    }

    // Find data chunk
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut num_channels = 0u16;

    while pos < data.len() - 8 {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size = u32::from_le_bytes([
            data[pos + 4],
            data[pos + 5],
            data[pos + 6],
            data[pos + 7],
        ]) as usize;

        if chunk_id == b"fmt " {
            num_channels = u16::from_le_bytes([data[pos + 10], data[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                data[pos + 12],
                data[pos + 13],
                data[pos + 14],
                data[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([data[pos + 22], data[pos + 23]]);
        } else if chunk_id == b"data" {
            let audio_data = &data[pos + 8..pos + 8 + chunk_size];

            // Convert to f32
            let samples: Vec<f32> = if bits_per_sample == 16 {
                audio_data
                    .chunks(2)
                    .map(|chunk| {
                        let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample as f32 / 32768.0
                    })
                    .collect()
            } else {
                return Err(candle::Error::Msg(
                    format!("Unsupported bits per sample: {}", bits_per_sample).into(),
                ));
            };

            // Convert to mono if stereo
            let mono: Vec<f32> = if num_channels == 2 {
                samples.chunks(2).map(|c| (c[0] + c[1]) / 2.0).collect()
            } else {
                samples
            };

            // Resample if needed
            if sample_rate as usize != target_sample_rate {
                return resample(&mono, sample_rate as usize, target_sample_rate);
            }

            return Ok(mono);
        }

        pos += 8 + chunk_size;
        if chunk_size % 2 == 1 {
            pos += 1; // Padding byte
        }
    }

    Err(candle::Error::Msg("No data chunk found in WAV file".into()))
}

/// Simple linear resampling
fn resample(samples: &[f32], from_rate: usize, to_rate: usize) -> Result<Vec<f32>> {
    let ratio = from_rate as f64 / to_rate as f64;
    let new_len = (samples.len() as f64 / ratio) as usize;
    let mut resampled = Vec::with_capacity(new_len);

    for i in 0..new_len {
        let src_idx = i as f64 * ratio;
        let src_floor = src_idx.floor() as usize;
        let frac = src_idx - src_floor as f64;

        let sample = if src_floor + 1 < samples.len() {
            samples[src_floor] * (1.0 - frac as f32) + samples[src_floor + 1] * frac as f32
        } else {
            samples[src_floor.min(samples.len() - 1)]
        };

        resampled.push(sample);
    }

    Ok(resampled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filterbank() {
        let processor = AudioProcessor::new(16000, 512, 160, 128, &Device::Cpu);
        let filters = processor.mel_filterbank();

        // Check dimensions
        assert_eq!(filters.len(), 128 * 257);

        // Filters should sum to approximately 1 in the middle
        let mid_sum: f32 = (0..257).map(|k| filters[64 * 257 + k]).sum();
        assert!(mid_sum > 0.0 && mid_sum < 3.0);
    }

    #[test]
    fn test_resample() {
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.001).sin()).collect();
        let resampled = resample(&samples, 16000, 8000).unwrap();

        // Should be half the length
        assert_eq!(resampled.len(), 8000);
    }
}
