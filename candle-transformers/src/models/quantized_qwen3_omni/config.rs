//! Configuration for quantized Qwen3-Omni model
//!
//! These configs match the GGUF tensor structure from qwen3-omni-q8_0.gguf

use candle_nn::Activation;

/// Top-level configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub audio_tower: AudioTowerConfig,
    pub thinker: ThinkerConfig,
    pub talker: TalkerConfig,
    pub code2wav: Code2WavConfig,
    pub vocab_size: usize,
    pub sample_rate: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            audio_tower: AudioTowerConfig::default(),
            thinker: ThinkerConfig::default(),
            talker: TalkerConfig::default(),
            code2wav: Code2WavConfig::default(),
            vocab_size: 152064,
            sample_rate: 24000,
        }
    }
}

/// Audio Tower configuration
/// Weights at: thinker.audio_tower.*
#[derive(Debug, Clone)]
pub struct AudioTowerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub conv_channels: usize,
    pub output_size: usize,
    pub rms_norm_eps: f64,
}

impl Default for AudioTowerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1280,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            intermediate_size: 5120,
            conv_channels: 480,
            output_size: 2048,
            rms_norm_eps: 1e-6,
        }
    }
}

/// Thinker (main reasoning model) configuration
/// Weights at: thinker.model.*
#[derive(Debug, Clone)]
pub struct ThinkerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub max_position_embeddings: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub hidden_act: Activation,
}

impl Default for ThinkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            head_dim: 128,
            moe_intermediate_size: 768,
            num_experts: 128,
            num_experts_per_tok: 8,
            norm_topk_prob: true,
            max_position_embeddings: 32768,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            vocab_size: 152064,
            hidden_act: Activation::Silu,
        }
    }
}

/// Talker (code_predictor speech synthesis model) configuration
/// Weights at: talker.code_predictor.*
///
/// GGUF structure:
/// - talker.code_predictor.model.codec_embedding.{0-14}.weight: [2048, 1024] Q8_0
/// - talker.code_predictor.model.layers.{N}.self_attn.{q,k,v,o}_proj.weight: Q8_0
/// - talker.code_predictor.model.layers.{N}.self_attn.{q,k}_norm.weight: F32
/// - talker.code_predictor.model.layers.{N}.mlp.{gate,up,down}_proj.weight: Q8_0
/// - talker.code_predictor.model.layers.{N}.{input,post_attention}_layernorm.weight: F32
/// - talker.code_predictor.lm_head.{0-14}.weight: [2048, 1024] Q8_0
#[derive(Debug, Clone)]
pub struct TalkerConfig {
    /// Hidden size of Talker model (1024)
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (GQA)
    pub num_key_value_heads: usize,
    /// Dimension per attention head
    pub head_dim: usize,
    /// MLP intermediate size
    pub intermediate_size: usize,
    /// Number of codebooks (15 for audio)
    pub num_codebooks: usize,
    /// Codebook vocabulary size (2048)
    pub codebook_size: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// Hidden activation function
    pub hidden_act: Activation,
    /// Max sequence length for RoPE
    pub max_position_embeddings: usize,
    /// RoPE theta base
    pub rope_theta: f64,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 5, // From GGUF: layers 0-4
            num_attention_heads: 16, // q_proj output 2048 / head_dim 128 = 16
            num_key_value_heads: 8, // k_proj output 1024 / head_dim 128 = 8
            head_dim: 128, // From q_norm weight size [128]
            intermediate_size: 3072, // From gate_proj [3072, 1024]
            num_codebooks: 15,
            codebook_size: 2048,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
            max_position_embeddings: 8192,
            rope_theta: 10000.0,
        }
    }
}

/// Code2Wav (vocoder) configuration
/// Weights at: code2wav.*
///
/// GGUF structure:
/// - code_embedding.weight: [1024, 32768] Q8
/// - pre_transformer.layers.N.*: 8 layers
/// - upsample.N.*: 2 blocks
/// - decoder.N.*: HiFi-GAN style blocks
#[derive(Debug, Clone)]
pub struct Code2WavConfig {
    /// Embedding dimension (1024)
    pub embedding_dim: usize,
    /// Number of codebooks (8 for code_embedding vocab = 32768 / 4096)
    pub num_codebooks: usize,
    /// Codebook vocabulary size per book (4096)
    pub codebook_size: usize,
    /// Number of pre-transformer layers
    pub num_transformer_layers: usize,
    /// Number of attention heads in pre-transformer
    pub num_attention_heads: usize,
    /// Intermediate size for MLP
    pub intermediate_size: usize,
    /// Number of upsample blocks
    pub num_upsample_blocks: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// Hidden activation
    pub hidden_act: Activation,
}

impl Default for Code2WavConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 1024,
            // 32768 / 4096 = 8 codebooks based on GGUF
            num_codebooks: 8,
            codebook_size: 4096,
            num_transformer_layers: 8,
            num_attention_heads: 8,
            intermediate_size: 3072, // 1024 * 3
            num_upsample_blocks: 2,
            rms_norm_eps: 1e-6,
            hidden_act: Activation::Silu,
        }
    }
}
