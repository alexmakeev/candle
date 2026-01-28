use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("=== wgpu VarBuilder test ===\n");

    #[cfg(not(feature = "wgpu"))]
    {
        println!("wgpu feature not enabled. Compile with --features wgpu");
        return Ok(());
    }

    #[cfg(feature = "wgpu")]
    {
        use candle_core::{safetensors::MmapedSafetensors, wgpu_backend};

        // Check if wgpu is available
        if !wgpu_backend::is_available() {
            println!("No wgpu adapters found!");
            return Ok(());
        }

        println!("Step 1: Create test tensors on CPU and save to safetensors...");

        // Create test tensors
        let cpu = Device::Cpu;
        let weight = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &cpu)?;
        let bias = Tensor::from_slice(&[0.1f32, 0.2, 0.3], (3,), &cpu)?;

        // Save to safetensors file
        let tmp_path = "/tmp/wgpu_test.safetensors";
        let mut tensors_map = HashMap::new();
        tensors_map.insert("weight".to_string(), weight.clone());
        tensors_map.insert("bias".to_string(), bias.clone());
        candle_core::safetensors::save(&tensors_map, tmp_path)?;
        println!("  Saved weight {:?} and bias {:?} to {}", weight.shape(), bias.shape(), tmp_path);

        println!("\nStep 2: Create WgpuDevice...");
        let device = Device::new_wgpu(0)?;
        println!("  Device: {:?}", device);

        println!("\nStep 3: Load safetensors to WgpuDevice...");
        let safetensors = unsafe { MmapedSafetensors::new(tmp_path)? };

        // Load weight tensor to wgpu
        let loaded_weight = safetensors.load("weight", &device)?;
        println!("  Loaded weight: {:?}", loaded_weight);

        // Verify by copying back to CPU
        let weight_cpu = loaded_weight.to_device(&Device::Cpu)?;
        let weight_values = weight_cpu.to_vec2::<f32>()?;
        println!("  Weight values: {:?}", weight_values);
        assert_eq!(weight_values, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);

        // Load bias tensor to wgpu
        let loaded_bias = safetensors.load("bias", &device)?;
        println!("  Loaded bias: {:?}", loaded_bias);

        let bias_cpu = loaded_bias.to_device(&Device::Cpu)?;
        let bias_values = bias_cpu.to_vec1::<f32>()?;
        println!("  Bias values: {:?}", bias_values);
        assert_eq!(bias_values, vec![0.1, 0.2, 0.3]);

        println!("\nStep 4: Test simple linear operation (matmul + bias)...");
        let input = Tensor::from_slice(&[1.0f32, 1.0], (1, 2), &device)?;

        // Linear: input @ weight + bias
        let output = input.matmul(&loaded_weight)?;
        let output = output.broadcast_add(&loaded_bias)?;

        println!("  Input: {:?}", input);
        println!("  Output: {:?}", output);

        let output_cpu = output.to_device(&Device::Cpu)?;
        let output_values = output_cpu.to_vec2::<f32>()?;
        println!("  Output values: {:?}", output_values);

        // Expected: [1, 1] @ [[1,2,3],[4,5,6]] + [0.1, 0.2, 0.3]
        // = [5, 7, 9] + [0.1, 0.2, 0.3] = [5.1, 7.2, 9.3]
        println!("  Expected: [[5.1, 7.2, 9.3]]");

        println!("\nStep 5: Test BF16 loading...");
        let bf16_weight = loaded_weight.to_dtype(DType::BF16)?;
        println!("  BF16 weight: {:?}", bf16_weight);

        let bf16_input = input.to_dtype(DType::BF16)?;
        let bf16_output = bf16_input.matmul(&bf16_weight)?;
        println!("  BF16 matmul output: {:?}", bf16_output);

        let bf16_output_f32 = bf16_output.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        println!("  BF16 output values: {:?}", bf16_output_f32.to_vec2::<f32>()?);

        // Cleanup
        std::fs::remove_file(tmp_path)?;

        println!("\n=== All VarBuilder tests passed! ===");
    }

    Ok(())
}
