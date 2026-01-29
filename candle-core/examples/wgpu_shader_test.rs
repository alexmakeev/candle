//! Minimal wgpu shader test — runs each operation individually and checks correctness.
//! Build: cargo run --release --features wgpu -p candle-core --example wgpu_shader_test

use candle_core::{DType, Device, Result, Tensor};

fn test_cast() -> Result<()> {
    let device = &Device::new_wgpu(0)?;
    // F32 → BF16 → F32 roundtrip
    let f32_data = vec![1.0f32, 2.0, 3.0, -4.0, 0.5, 100.0, -0.001, 65504.0];
    let t = Tensor::from_vec(f32_data.clone(), (8,), device)?;
    let bf16 = t.to_dtype(DType::BF16)?;
    let back = bf16.to_dtype(DType::F32)?;
    let result: Vec<f32> = back.to_vec1()?;
    for (i, (orig, got)) in f32_data.iter().zip(result.iter()).enumerate() {
        let diff = (orig - got).abs();
        // BF16 has ~0.4% relative error for normal values
        let tol = orig.abs() * 0.01 + 0.01;
        assert!(diff < tol, "cast[{i}]: expected {orig}, got {got}, diff={diff}");
    }
    eprintln!("[OK] cast F32<->BF16");
    Ok(())
}

fn test_binary() -> Result<()> {
    let device = &Device::new_wgpu(0)?;
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

fn test_unary() -> Result<()> {
    let device = &Device::new_wgpu(0)?;
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

fn test_matmul_bf16() -> Result<()> {
    let device = &Device::new_wgpu(0)?;
    // Simple 2x2 matmul
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2, 2), device)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2, 2), device)?.to_dtype(DType::BF16)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = c.to_vec2::<f32>()?.into_iter().flatten().collect();
    // [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    let expected = vec![19.0, 22.0, 43.0, 50.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 1.0, "matmul[{i}]: expected {e}, got {g}");
    }

    eprintln!("[OK] matmul BF16");
    Ok(())
}

fn test_index_select() -> Result<()> {
    let device = &Device::new_wgpu(0)?;
    // 4 rows of 2 elements each (even row_size for BF16 packing)
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

fn test_reduce() -> Result<()> {
    let device = &Device::new_wgpu(0)?;
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        device,
    )?.to_dtype(DType::BF16)?;

    // Sum along last dim
    let sum = data.sum_keepdim(1)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = sum.to_vec2::<f32>()?.into_iter().flatten().collect();
    let expected = vec![6.0, 15.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.5, "reduce_sum[{i}]: expected {e}, got {g}");
    }

    eprintln!("[OK] reduce BF16 Sum");
    Ok(())
}

fn test_affine() -> Result<()> {
    let device = &Device::new_wgpu(0)?;
    let data = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), device)?.to_dtype(DType::BF16)?;
    let result = data.affine(2.0, 1.0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let expected = vec![3.0, 5.0, 7.0, 9.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.1, "affine[{i}]: expected {e}, got {g}");
    }

    eprintln!("[OK] affine BF16");
    Ok(())
}

fn test_copy_strided() -> Result<()> {
    let device = &Device::new_wgpu(0)?;
    // Create a tensor and transpose it (makes it non-contiguous)
    let data = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
        (2, 3),
        device,
    )?.to_dtype(DType::BF16)?;
    // Transpose makes it non-contiguous
    let transposed = data.t()?;
    // Force contiguous copy
    let contiguous = transposed.contiguous()?;
    let result: Vec<f32> = contiguous.to_dtype(DType::F32)?.to_vec2::<f32>()?.into_iter().flatten().collect();
    // Transpose of [[1,2,3],[4,5,6]] = [[1,4],[2,5],[3,6]]
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

    let tests: Vec<(&str, fn() -> Result<()>)> = vec![
        ("cast", test_cast),
        ("binary", test_binary),
        ("unary", test_unary),
        ("matmul_bf16", test_matmul_bf16),
        ("index_select", test_index_select),
        ("reduce", test_reduce),
        ("affine", test_affine),
        ("copy_strided", test_copy_strided),
    ];

    let mut passed = 0;
    let mut failed = 0;

    for (name, test_fn) in &tests {
        match test_fn() {
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
