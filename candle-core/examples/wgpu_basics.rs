use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn main() -> Result<()> {
    println!("=== wgpu backend basic tests ===\n");

    // Check if wgpu is available
    #[cfg(not(feature = "wgpu"))]
    {
        println!("wgpu feature not enabled. Compile with --features wgpu");
        return Ok(());
    }

    #[cfg(feature = "wgpu")]
    {
        use candle_core::wgpu_backend;

        // List available adapters
        println!("Checking wgpu availability...");
        if !wgpu_backend::is_available() {
            println!("No wgpu adapters found!");
            return Ok(());
        }

        let adapters = wgpu_backend::list_adapters();
        println!("Found {} adapter(s):", adapters.len());
        for (i, info) in adapters.iter().enumerate() {
            println!("  [{i}] {:?}", info);
        }
        println!();

        // Create device
        println!("Creating WgpuDevice(0)...");
        let device = Device::new_wgpu(0)?;
        println!("Device created: {:?}\n", device);

        // Test 1: Create zeros tensor
        println!("Test 1: zeros tensor");
        let zeros = Tensor::zeros((2, 3), DType::F32, &device)?;
        println!("  zeros((2, 3), F32): {:?}", zeros);
        let zeros_cpu = zeros.to_device(&Device::Cpu)?;
        println!("  values: {:?}", zeros_cpu.to_vec2::<f32>()?);
        println!();

        // Test 2: Create ones tensor
        println!("Test 2: ones tensor");
        let ones = Tensor::ones((2, 3), DType::F32, &device)?;
        println!("  ones((2, 3), F32): {:?}", ones);
        let ones_cpu = ones.to_device(&Device::Cpu)?;
        println!("  values: {:?}", ones_cpu.to_vec2::<f32>()?);
        println!();

        // Test 3: Create tensor from slice
        println!("Test 3: from_slice");
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_slice(&data, (2, 3), &device)?;
        println!("  from_slice([1,2,3,4,5,6], (2,3)): {:?}", t);
        let t_cpu = t.to_device(&Device::Cpu)?;
        println!("  values: {:?}", t_cpu.to_vec2::<f32>()?);
        println!();

        // Test 4: Add tensors
        println!("Test 4: add");
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let b = Tensor::from_slice(&[10.0f32, 20.0, 30.0, 40.0], (2, 2), &device)?;
        let c = a.add(&b)?;
        println!("  a + b: {:?}", c);
        let c_cpu = c.to_device(&Device::Cpu)?;
        println!("  values: {:?}", c_cpu.to_vec2::<f32>()?);
        println!();

        // Test 5: Matmul
        println!("Test 5: matmul");
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?;
        let c = a.matmul(&b)?;
        println!("  [[1,2],[3,4]] @ [[1,2],[3,4]]: {:?}", c);
        let c_cpu = c.to_device(&Device::Cpu)?;
        println!("  values: {:?}", c_cpu.to_vec2::<f32>()?);
        println!("  expected: [[7, 10], [15, 22]]");
        println!();

        // Test 6: BF16 support
        println!("Test 6: BF16 tensor");
        let bf16 = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?
            .to_dtype(DType::BF16)?;
        println!("  bf16 tensor: {:?}", bf16);
        let bf16_f32 = bf16.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        println!("  values (as f32): {:?}", bf16_f32.to_vec2::<f32>()?);
        println!();

        // Test 7: BF16 matmul
        println!("Test 7: BF16 matmul");
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?
            .to_dtype(DType::BF16)?;
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], (2, 2), &device)?
            .to_dtype(DType::BF16)?;
        let c = a.matmul(&b)?;
        println!("  bf16 matmul result: {:?}", c);
        let c_cpu = c.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        println!("  values (as f32): {:?}", c_cpu.to_vec2::<f32>()?);
        println!("  expected: [[7, 10], [15, 22]]");
        println!();

        // Test 8: Larger matmul benchmark
        println!("Test 8: Larger matmul benchmark (512x512)");
        let a = Tensor::randn(0f32, 1.0, (512, 512), &device)?;
        let b = Tensor::randn(0f32, 1.0, (512, 512), &device)?;

        // Warmup
        let _ = a.matmul(&b)?;

        // Benchmark
        let start = std::time::Instant::now();
        for _ in 0..10 {
            let _ = a.matmul(&b)?;
        }
        let elapsed = start.elapsed();
        println!("  10x matmul(512x512): {:?} ({:?} per op)", elapsed, elapsed / 10);
        println!();

        println!("=== All tests passed! ===");
    }

    Ok(())
}
