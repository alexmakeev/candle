//! Minimal wgpu shader test — runs each operation individually and checks correctness.
//! Build: cargo run --release --features wgpu -p candle-core --example wgpu_shader_test
//!
//! Usage:
//!   WGPU_GPU_ALL=0 ./wgpu_shader_test              # all CPU
//!   WGPU_GPU_ALL=1 ./wgpu_shader_test              # all GPU
//!   WGPU_GPU_CAST=1 ./wgpu_shader_test             # only cast on GPU
//!   WGPU_GPU_ALL=1 ./wgpu_shader_test copy_strided  # run only copy_strided test

use candle_core::{DType, Device, Result, Tensor};

fn test_cast(device: &Device) -> Result<()> {
    let f32_data = vec![1.0f32, 2.0, 3.0, -4.0, 0.5, 100.0, -0.001, 65504.0];
    let t = Tensor::from_vec(f32_data.clone(), (8,), device)?;
    let bf16 = t.to_dtype(DType::BF16)?;
    let back = bf16.to_dtype(DType::F32)?;
    let result: Vec<f32> = back.to_vec1()?;
    for (i, (orig, got)) in f32_data.iter().zip(result.iter()).enumerate() {
        let diff = (orig - got).abs();
        let tol = orig.abs() * 0.01 + 0.01;
        assert!(diff < tol, "cast[{i}]: expected {orig}, got {got}, diff={diff}");
    }
    eprintln!("[OK] cast F32<->BF16");
    Ok(())
}

fn test_binary(device: &Device) -> Result<()> {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), device)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (4,), device)?.to_dtype(DType::BF16)?;

    let sum = (&a + &b)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let expected_sum = vec![6.0, 8.0, 10.0, 12.0];
    for (i, (e, g)) in expected_sum.iter().zip(sum.iter()).enumerate() {
        assert!((e - g).abs() < 0.1, "add[{i}]: expected {e}, got {g}");
    }

    let prod = (&a * &b)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let expected_prod = vec![5.0, 12.0, 21.0, 32.0];
    for (i, (e, g)) in expected_prod.iter().zip(prod.iter()).enumerate() {
        assert!((e - g).abs() < 0.5, "mul[{i}]: expected {e}, got {g}");
    }

    eprintln!("[OK] binary BF16 (add, mul)");
    Ok(())
}

fn test_unary(device: &Device) -> Result<()> {
    let a = Tensor::from_vec(vec![0.0f32, 1.0, -1.0, 2.0], (4,), device)?.to_dtype(DType::BF16)?;
    let neg = a.neg()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let expected = vec![0.0, -1.0, 1.0, -2.0];
    for (i, (e, g)) in expected.iter().zip(neg.iter()).enumerate() {
        assert!((e - g).abs() < 0.01, "neg[{i}]: expected {e}, got {g}");
    }

    let b = Tensor::from_vec(vec![1.0f32, 2.0, 0.5, 0.1], (4,), device)?.to_dtype(DType::BF16)?;
    let exp_result = b.exp()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let expected_exp = vec![2.718, 7.389, 1.649, 1.105];
    for (i, (e, g)) in expected_exp.iter().zip(exp_result.iter()).enumerate() {
        assert!((e - g).abs() < 0.1, "exp[{i}]: expected {e}, got {g}");
    }

    eprintln!("[OK] unary BF16 (neg, exp)");
    Ok(())
}

fn test_matmul_bf16(device: &Device) -> Result<()> {
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), device)?.to_dtype(DType::BF16)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = c.to_vec2::<f32>()?.into_iter().flatten().collect();
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 1.0, "matmul[{i}]: expected {e}, got {g}");
    }
    eprintln!("[OK] matmul BF16");
    Ok(())
}

fn test_index_select(device: &Device) -> Result<()> {
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        (4, 2),
        device,
    )?.to_dtype(DType::BF16)?;
    let ids = Tensor::from_vec(vec![0u32, 2, 3], (3,), device)?;
    let selected = data.index_select(&ids, 0)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = selected.to_vec2::<f32>()?.into_iter().flatten().collect();
    let expected = vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.1, "index_select[{i}]: expected {e}, got {g}");
    }
    eprintln!("[OK] index_select BF16 dim=0");
    Ok(())
}

fn test_reduce(device: &Device) -> Result<()> {
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        device,
    )?.to_dtype(DType::BF16)?;
    let sum = data.sum_keepdim(1)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = sum.to_vec2::<f32>()?.into_iter().flatten().collect();
    let expected = vec![6.0, 15.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.5, "reduce_sum[{i}]: expected {e}, got {g}");
    }
    eprintln!("[OK] reduce BF16 Sum");
    Ok(())
}

fn test_affine(device: &Device) -> Result<()> {
    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), device)?.to_dtype(DType::BF16)?;
    let result = data.affine(2.0, 1.0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let expected = vec![3.0, 5.0, 7.0, 9.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.1, "affine[{i}]: expected {e}, got {g}");
    }
    eprintln!("[OK] affine BF16");
    Ok(())
}

fn test_copy_strided_f32(device: &Device) -> Result<()> {
    // Test F32 strided copy (transpose -> contiguous) directly
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        device,
    )?;
    let transposed = data.t()?;
    let contiguous = transposed.contiguous()?;
    let result: Vec<f32> = contiguous.to_vec2::<f32>()?.into_iter().flatten().collect();
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.01, "copy_strided_f32[{i}]: expected {e}, got {g}");
    }
    eprintln!("[OK] copy_strided F32 (transpose->contiguous)");
    Ok(())
}

fn test_odd_count(device: &Device) -> Result<()> {
    // Test BF16 with odd element count (padding to u32 boundary)
    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0], (5,), device)?
        .to_dtype(DType::BF16)?;
    let result: Vec<f32> = data.to_dtype(DType::F32)?.to_vec1()?;
    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.1, "odd_count[{i}]: expected {e}, got {g}");
    }

    // Matmul with odd K dimension
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), device)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(vec![4.0f32, 5.0, 6.0], (3, 1), device)?.to_dtype(DType::BF16)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = c.to_vec2::<f32>()?.into_iter().flatten().collect();
    // 1*4 + 2*5 + 3*6 = 32
    assert!((result[0] - 32.0).abs() < 1.0, "odd_matmul: expected 32, got {}", result[0]);

    eprintln!("[OK] odd element count BF16");
    Ok(())
}

fn test_copy_strided(device: &Device) -> Result<()> {
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        device,
    )?.to_dtype(DType::BF16)?;
    let transposed = data.t()?;
    let contiguous = transposed.contiguous()?;
    let result: Vec<f32> = contiguous.to_dtype(DType::F32)?.to_vec2::<f32>()?.into_iter().flatten().collect();
    let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.1, "copy_strided[{i}]: expected {e}, got {g}");
    }
    eprintln!("[OK] copy_strided BF16 (transpose->contiguous)");
    Ok(())
}

fn main() {
    eprintln!("=== WGPU Shader Test ===");
    let flags = candle_core::shader_flags();
    eprintln!("Flags: {:?}", flags);
    eprintln!();

    let device = Device::new_wgpu(0).expect("Failed to create wgpu device");

    let filter = std::env::args().nth(1);

    let tests: Vec<(&str, Box<dyn Fn(&Device) -> Result<()>>)> = vec![
        ("cast", Box::new(test_cast)),
        ("binary", Box::new(test_binary)),
        ("unary", Box::new(test_unary)),
        ("matmul_bf16", Box::new(test_matmul_bf16)),
        ("index_select", Box::new(test_index_select)),
        ("reduce", Box::new(test_reduce)),
        ("affine", Box::new(test_affine)),
        ("copy_strided_f32", Box::new(test_copy_strided_f32)),
        ("copy_strided", Box::new(test_copy_strided)),
        ("odd_count", Box::new(test_odd_count)),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (name, test_fn) in &tests {
        if let Some(ref f) = filter {
            if !name.contains(f.as_str()) {
                continue;
            }
        }
        eprintln!("--- {name} ---");
        match test_fn(&device) {
            Ok(()) => passed += 1,
            Err(e) => {
                eprintln!("[FAIL] {name}: {e}");
                failed += 1;
            }
        }
    }

    eprintln!();
    eprintln!("=== Results: {passed} passed, {failed} failed ===");
    if failed > 0 {
        std::process::exit(1);
    }
}
