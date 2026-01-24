//! Streaming quantizer for large models
//!
//! Processes tensors one at a time to minimize memory usage.
//! Peak memory: ~2x largest tensor size instead of full model.

use byteorder::{LittleEndian, WriteBytesExt};
use candle::quantized::{GgmlDType, QTensor};
use candle::{Device, Result};
use std::io::{Seek, Write};
use std::path::Path;

/// Tensor metadata for GGUF header
struct TensorMeta {
    name: String,
    shape: Vec<usize>,
    dtype: GgmlDType,
    size_in_bytes: usize,
}

/// Quantize safetensors to GGUF with streaming (low memory usage)
///
/// Strategy:
/// 1. Memory-map all input files (no data loaded)
/// 2. Collect tensor metadata
/// 3. Write GGUF header
/// 4. Process tensors one by one: load → quantize → write → drop
pub fn streaming_quantize(
    in_files: &[std::path::PathBuf],
    out_file: &Path,
    target_dtype: GgmlDType,
) -> Result<()> {
    println!("Opening {} input files with memory mapping...", in_files.len());

    // Open all files with mmap (metadata only, no data loaded)
    let safetensors = unsafe { candle::safetensors::MmapedSafetensors::multi(in_files)? };

    // Collect all tensor views (still no data loaded)
    let tensor_views = safetensors.tensors();
    println!("Found {} tensors", tensor_views.len());

    // Calculate metadata and target sizes
    let block_size = target_dtype.block_size();
    let mut tensor_metas: Vec<TensorMeta> = Vec::with_capacity(tensor_views.len());

    for (name, view) in &tensor_views {
        let shape: Vec<usize> = view.shape().to_vec();
        let elem_count: usize = shape.iter().product();

        // Decide quantization dtype
        // Must be 2D AND last dim divisible by block_size for quantization
        let last_dim = shape.last().copied().unwrap_or(0);
        let can_quantize = shape.len() == 2
            && last_dim % block_size == 0
            && elem_count % block_size == 0;

        let (dtype, size_in_bytes) = if can_quantize {
            // 2D tensor with compatible shape: quantize
            let num_blocks = elem_count / block_size;
            let size = num_blocks * target_dtype.type_size();
            (target_dtype, size)
        } else {
            // 1D or incompatible dimensions: keep as F32
            (GgmlDType::F32, elem_count * 4)
        };

        tensor_metas.push(TensorMeta {
            name: name.clone(),
            shape,
            dtype,
            size_in_bytes,
        });
    }

    // Sort by name for deterministic output
    tensor_metas.sort_by(|a, b| a.name.cmp(&b.name));

    // Calculate offsets
    let mut offset = 0usize;
    let mut offsets: Vec<usize> = Vec::with_capacity(tensor_metas.len());
    for meta in &tensor_metas {
        offsets.push(offset);
        let padding = 31 - (31 + meta.size_in_bytes) % 32;
        offset += meta.size_in_bytes + padding;
    }

    println!("Writing GGUF file: {:?}", out_file);
    let mut out = std::fs::File::create(out_file)?;

    // Write GGUF header
    write_gguf_header(&mut out, &tensor_metas, &offsets)?;

    // Process tensors one by one
    let device = Device::Cpu;
    let total = tensor_metas.len();

    for (idx, meta) in tensor_metas.iter().enumerate() {
        print!("\r[{}/{}] Quantizing: {} {:?} -> {:?}",
               idx + 1, total, meta.name, meta.shape, meta.dtype);
        std::io::stdout().flush()?;

        // Load this tensor (actual data load happens here)
        let tensor = safetensors.load(&meta.name, &device)?;

        // Quantize
        let qtensor = QTensor::quantize(&tensor, meta.dtype)?;

        // Write tensor data with padding
        let data = qtensor.data()?;
        out.write_all(&data)?;

        let padding = 31 - (31 + data.len()) % 32;
        if padding > 0 {
            out.write_all(&vec![0u8; padding])?;
        }

        // Tensor goes out of scope here, memory freed
    }

    println!("\n✅ Done! Output: {:?}", out_file);
    Ok(())
}

/// Convert GgmlDType to GGUF type number
fn dtype_to_u32(dtype: GgmlDType) -> u32 {
    match dtype {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::Q4_0 => 2,
        GgmlDType::Q4_1 => 3,
        GgmlDType::Q5_0 => 6,
        GgmlDType::Q5_1 => 7,
        GgmlDType::Q8_0 => 8,
        GgmlDType::Q8_1 => 9,
        GgmlDType::Q2K => 10,
        GgmlDType::Q3K => 11,
        GgmlDType::Q4K => 12,
        GgmlDType::Q5K => 13,
        GgmlDType::Q6K => 14,
        GgmlDType::Q8K => 15,
        GgmlDType::BF16 => 30,
    }
}

fn write_gguf_header<W: Write + Seek>(
    w: &mut W,
    tensors: &[TensorMeta],
    offsets: &[usize],
) -> Result<()> {
    // Magic and version
    w.write_u32::<LittleEndian>(0x46554747)?; // GGUF magic
    w.write_u32::<LittleEndian>(2)?; // version 2

    // Tensor count and metadata count
    w.write_u64::<LittleEndian>(tensors.len() as u64)?;
    w.write_u64::<LittleEndian>(0)?; // no metadata

    // Write tensor infos
    for (meta, &offset) in tensors.iter().zip(offsets.iter()) {
        // Name
        w.write_u64::<LittleEndian>(meta.name.len() as u64)?;
        w.write_all(meta.name.as_bytes())?;

        // Dimensions
        w.write_u32::<LittleEndian>(meta.shape.len() as u32)?;
        for &dim in meta.shape.iter().rev() {
            w.write_u64::<LittleEndian>(dim as u64)?;
        }

        // Type and offset
        w.write_u32::<LittleEndian>(dtype_to_u32(meta.dtype))?;
        w.write_u64::<LittleEndian>(offset as u64)?;
    }

    // Padding to alignment
    let pos = w.stream_position()? as usize;
    let padding = 31 - (31 + pos) % 32;
    w.write_all(&vec![0u8; padding])?;

    Ok(())
}

/// Get model size estimate after quantization
pub fn estimate_size(in_files: &[std::path::PathBuf], target_dtype: GgmlDType) -> Result<(usize, usize)> {
    let safetensors = unsafe { candle::safetensors::MmapedSafetensors::multi(in_files)? };
    let tensor_views = safetensors.tensors();

    let block_size = target_dtype.block_size();
    let mut original_size = 0usize;
    let mut quantized_size = 0usize;

    for (_name, view) in &tensor_views {
        let shape: Vec<usize> = view.shape().to_vec();
        let elem_count: usize = shape.iter().product();

        // Original size (assume BF16)
        original_size += elem_count * 2;

        // Quantized size
        let last_dim = shape.last().copied().unwrap_or(0);
        let can_quantize = shape.len() == 2
            && last_dim % block_size == 0
            && elem_count % block_size == 0;

        if can_quantize {
            let num_blocks = elem_count / block_size;
            quantized_size += num_blocks * target_dtype.type_size();
        } else {
            quantized_size += elem_count * 4; // F32 for non-quantized
        }
    }

    Ok((original_size, quantized_size))
}
