//! Configuration for Qwen3-Omni model components

use candle_nn::Activation;
use serde::Deserialize;

/// Top-level configuration for Qwen3-Omni
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub model_type: String,

    /// AuT encoder configuration
    #[serde(default)]
    pub aut_encoder: AuTEncoderConfig,

    /// Thinker (main reasoning) configuration
    #[serde(default)]
    pub thinker: ThinkerConfig,

    /// Talker (speech synthesis) configuration
    #[serde(default)]
    pub talker: TalkerConfig,

    /// Code2Wav (vocoder) configuration
    #[serde(default)]
    pub code2wav: Code2WavConfig,

    /// Vocabulary size for text tokens
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Sample rate for audio (default 16kHz)
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
}

fn default_vocab_size() -> usize {
    151936
}

fn default_sample_rate() -> usize {
    16000
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_type: "qwen3_omni".to_string(),
            aut_encoder: AuTEncoderConfig::default(),
            thinker: ThinkerConfig::default(),
            talker: TalkerConfig::default(),
            code2wav: Code2WavConfig::default(),
            vocab_size: default_vocab_size(),
            sample_rate: default_sample_rate(),
        }
    }
}

/// AuT (Audio-to-Token) Encoder configuration
/// Custom audio encoder, NOT Whisper-based
#[derive(Debug, Clone, Deserialize)]
pub struct AuTEncoderConfig {
    /// Hidden dimension
    #[serde(default = "aut_hidden_size")]
    pub hidden_size: usize,

    /// Number of encoder layers
    #[serde(default = "aut_num_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads
    #[serde(default = "aut_num_heads")]
    pub num_attention_heads: usize,

    /// Audio frame size (samples per frame)
    #[serde(default = "aut_frame_size")]
    pub frame_size: usize,

    /// Audio hop size (stride between frames)
    #[serde(default = "aut_hop_size")]
    pub hop_size: usize,

    /// Number of mel filterbanks
    #[serde(default = "aut_n_mels")]
    pub n_mels: usize,

    /// FFT size for spectrogram
    #[serde(default = "aut_n_fft")]
    pub n_fft: usize,

    /// Output vocabulary size for audio tokens
    #[serde(default = "aut_audio_vocab_size")]
    pub audio_vocab_size: usize,

    /// RMS norm epsilon
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
}

fn aut_hidden_size() -> usize {
    1024
}

fn aut_num_layers() -> usize {
    24
}

fn aut_num_heads() -> usize {
    16
}

fn aut_frame_size() -> usize {
    400 // 25ms at 16kHz
}

fn aut_hop_size() -> usize {
    160 // 10ms at 16kHz
}

fn aut_n_mels() -> usize {
    128
}

fn aut_n_fft() -> usize {
    512
}

fn aut_audio_vocab_size() -> usize {
    4096
}

fn default_rms_eps() -> f64 {
    1e-6
}

impl Default for AuTEncoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: aut_hidden_size(),
            num_hidden_layers: aut_num_layers(),
            num_attention_heads: aut_num_heads(),
            frame_size: aut_frame_size(),
            hop_size: aut_hop_size(),
            n_mels: aut_n_mels(),
            n_fft: aut_n_fft(),
            audio_vocab_size: aut_audio_vocab_size(),
            rms_norm_eps: default_rms_eps(),
        }
    }
}

/// Thinker (main reasoning model) configuration
/// 30B MoE with 3B active parameters
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfig {
    /// Hidden dimension
    #[serde(default = "thinker_hidden_size")]
    pub hidden_size: usize,

    /// Number of transformer layers
    #[serde(default = "thinker_num_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads
    #[serde(default = "thinker_num_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads (GQA)
    #[serde(default = "thinker_num_kv_heads")]
    pub num_key_value_heads: usize,

    /// Head dimension
    #[serde(default = "thinker_head_dim")]
    pub head_dim: usize,

    /// MLP intermediate dimension
    #[serde(default = "thinker_intermediate_size")]
    pub intermediate_size: usize,

    /// MoE intermediate dimension
    #[serde(default = "thinker_moe_intermediate_size")]
    pub moe_intermediate_size: usize,

    /// Total number of experts
    #[serde(default = "thinker_num_experts")]
    pub num_experts: usize,

    /// Number of active experts per token
    #[serde(default = "thinker_experts_per_tok")]
    pub num_experts_per_tok: usize,

    /// Decoder sparse step (which layers use MoE)
    #[serde(default = "thinker_sparse_step")]
    pub decoder_sparse_step: usize,

    /// Whether to normalize top-k probabilities
    #[serde(default = "thinker_norm_topk")]
    pub norm_topk_prob: bool,

    /// Maximum position embeddings
    #[serde(default = "thinker_max_pos")]
    pub max_position_embeddings: usize,

    /// RoPE theta base
    #[serde(default = "thinker_rope_theta")]
    pub rope_theta: f64,

    /// RMS norm epsilon
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,

    /// Vocabulary size (shared with main config)
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Whether to tie embeddings
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Hidden activation function
    #[serde(default = "default_activation")]
    pub hidden_act: Activation,
}

fn thinker_hidden_size() -> usize {
    4096
}

fn thinker_num_layers() -> usize {
    40
}

fn thinker_num_heads() -> usize {
    32
}

fn thinker_num_kv_heads() -> usize {
    8
}

fn thinker_head_dim() -> usize {
    128
}

fn thinker_intermediate_size() -> usize {
    11008
}

fn thinker_moe_intermediate_size() -> usize {
    2816
}

fn thinker_num_experts() -> usize {
    64
}

fn thinker_experts_per_tok() -> usize {
    4
}

fn thinker_sparse_step() -> usize {
    2
}

fn thinker_norm_topk() -> bool {
    true
}

fn thinker_max_pos() -> usize {
    32768
}

fn thinker_rope_theta() -> f64 {
    1_000_000.0
}

fn default_activation() -> Activation {
    Activation::Silu
}

impl Default for ThinkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: thinker_hidden_size(),
            num_hidden_layers: thinker_num_layers(),
            num_attention_heads: thinker_num_heads(),
            num_key_value_heads: thinker_num_kv_heads(),
            head_dim: thinker_head_dim(),
            intermediate_size: thinker_intermediate_size(),
            moe_intermediate_size: thinker_moe_intermediate_size(),
            num_experts: thinker_num_experts(),
            num_experts_per_tok: thinker_experts_per_tok(),
            decoder_sparse_step: thinker_sparse_step(),
            norm_topk_prob: thinker_norm_topk(),
            max_position_embeddings: thinker_max_pos(),
            rope_theta: thinker_rope_theta(),
            rms_norm_eps: default_rms_eps(),
            vocab_size: default_vocab_size(),
            tie_word_embeddings: false,
            hidden_act: default_activation(),
        }
    }
}

/// Talker (speech synthesis model) configuration
/// 3B MoE with 0.3B active parameters
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    /// Hidden dimension
    #[serde(default = "talker_hidden_size")]
    pub hidden_size: usize,

    /// Number of transformer layers
    #[serde(default = "talker_num_layers")]
    pub num_hidden_layers: usize,

    /// Number of attention heads
    #[serde(default = "talker_num_heads")]
    pub num_attention_heads: usize,

    /// Number of key-value heads (GQA)
    #[serde(default = "talker_num_kv_heads")]
    pub num_key_value_heads: usize,

    /// Head dimension
    #[serde(default = "talker_head_dim")]
    pub head_dim: usize,

    /// MoE intermediate dimension
    #[serde(default = "talker_moe_intermediate_size")]
    pub moe_intermediate_size: usize,

    /// Total number of experts
    #[serde(default = "talker_num_experts")]
    pub num_experts: usize,

    /// Number of active experts per token
    #[serde(default = "talker_experts_per_tok")]
    pub num_experts_per_tok: usize,

    /// Number of codec codebooks
    #[serde(default = "talker_num_codebooks")]
    pub num_codebooks: usize,

    /// Codebook vocabulary size
    #[serde(default = "talker_codebook_size")]
    pub codebook_size: usize,

    /// RMS norm epsilon
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,

    /// Hidden activation function
    #[serde(default = "default_activation")]
    pub hidden_act: Activation,
}

fn talker_hidden_size() -> usize {
    1536
}

fn talker_num_layers() -> usize {
    24
}

fn talker_num_heads() -> usize {
    12
}

fn talker_num_kv_heads() -> usize {
    4
}

fn talker_head_dim() -> usize {
    128
}

fn talker_moe_intermediate_size() -> usize {
    1024
}

fn talker_num_experts() -> usize {
    32
}

fn talker_experts_per_tok() -> usize {
    4
}

fn talker_num_codebooks() -> usize {
    4
}

fn talker_codebook_size() -> usize {
    2048
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: talker_hidden_size(),
            num_hidden_layers: talker_num_layers(),
            num_attention_heads: talker_num_heads(),
            num_key_value_heads: talker_num_kv_heads(),
            head_dim: talker_head_dim(),
            moe_intermediate_size: talker_moe_intermediate_size(),
            num_experts: talker_num_experts(),
            num_experts_per_tok: talker_experts_per_tok(),
            num_codebooks: talker_num_codebooks(),
            codebook_size: talker_codebook_size(),
            rms_norm_eps: default_rms_eps(),
            hidden_act: default_activation(),
        }
    }
}

/// Code2Wav (neural vocoder) configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Code2WavConfig {
    /// Hidden dimension for vocoder
    #[serde(default = "code2wav_hidden_size")]
    pub hidden_size: usize,

    /// Number of upsampling layers
    #[serde(default = "code2wav_num_upsample")]
    pub num_upsample_layers: usize,

    /// Upsample rates per layer
    #[serde(default = "code2wav_upsample_rates")]
    pub upsample_rates: Vec<usize>,

    /// Number of residual blocks per layer
    #[serde(default = "code2wav_num_residual")]
    pub num_residual_blocks: usize,

    /// Number of codebooks (must match Talker)
    #[serde(default = "talker_num_codebooks")]
    pub num_codebooks: usize,

    /// Codebook vocabulary size (must match Talker)
    #[serde(default = "talker_codebook_size")]
    pub codebook_size: usize,
}

fn code2wav_hidden_size() -> usize {
    512
}

fn code2wav_num_upsample() -> usize {
    4
}

fn code2wav_upsample_rates() -> Vec<usize> {
    vec![8, 5, 4, 2] // Total 320x: 50Hz codec -> 16kHz audio
}

fn code2wav_num_residual() -> usize {
    3
}

impl Default for Code2WavConfig {
    fn default() -> Self {
        Self {
            hidden_size: code2wav_hidden_size(),
            num_upsample_layers: code2wav_num_upsample(),
            upsample_rates: code2wav_upsample_rates(),
            num_residual_blocks: code2wav_num_residual(),
            num_codebooks: talker_num_codebooks(),
            codebook_size: talker_codebook_size(),
        }
    }
}
