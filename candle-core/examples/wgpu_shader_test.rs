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

fn test_linear_matmul(device: &Device) -> Result<()> {
    // Simulate Linear layer: x @ w.t() where w is [out_dim, in_dim]
    // This tests that pre-transposed contiguous weights work correctly
    let x = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), device)?.to_dtype(DType::BF16)?;
    let w = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0,  // row 0
             0.0, 1.0, 0.0],    // row 1
        (2, 3),
        device,
    )?.to_dtype(DType::BF16)?;

    // Method 1: w.t() (non-contiguous) — the old way
    let wt = w.t()?;
    assert!(!wt.is_contiguous(), "transposed weight should be non-contiguous");
    let r1 = x.matmul(&wt)?.to_dtype(DType::F32)?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<_>>();

    // Method 2: w.t().contiguous() (contiguous) — the new way via weight_t
    let wt_contig = wt.contiguous()?;
    assert!(wt_contig.is_contiguous(), "contiguous weight should be contiguous");
    let r2 = x.matmul(&wt_contig)?.to_dtype(DType::F32)?.to_vec2::<f32>()?.into_iter().flatten().collect::<Vec<_>>();

    // Expected: x @ w.t() = [1,2,3] @ [[1,0],[0,1],[0,0]] = [1, 2]
    let expected = vec![1.0, 2.0];
    for (i, (e, g)) in expected.iter().zip(r1.iter()).enumerate() {
        assert!((e - g).abs() < 0.5, "linear_non_contig[{i}]: expected {e}, got {g}");
    }
    for (i, (e, g)) in expected.iter().zip(r2.iter()).enumerate() {
        assert!((e - g).abs() < 0.5, "linear_contig[{i}]: expected {e}, got {g}");
    }
    // Both methods must produce same result
    for (i, (a, b)) in r1.iter().zip(r2.iter()).enumerate() {
        assert!((a - b).abs() < 0.01, "linear_match[{i}]: non_contig={a}, contig={b}");
    }

    eprintln!("[OK] linear matmul BF16 (non-contiguous vs contiguous weight)");
    Ok(())
}

fn test_gemv_bf16(device: &Device) -> Result<()> {
    // Test 1: Simple GEMV — [1,4] @ [4,3] = [1,3]
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (1, 4), device)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0,
             1.0, 1.0, 1.0],
        (4, 3),
        device,
    )?.to_dtype(DType::BF16)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = c.to_vec2::<f32>()?.into_iter().flatten().collect();
    // [1,2,3,4] @ [[1,0,0],[0,1,0],[0,0,1],[1,1,1]] = [1+4, 2+4, 3+4] = [5, 6, 7]
    let expected = vec![5.0, 6.0, 7.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.5, "gemv_simple[{i}]: expected {e}, got {g}");
    }

    // Test 2: Larger K dimension — [1,128] @ [128,64] = [1,64]
    let a_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..128*64).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let a = Tensor::from_vec(a_data.clone(), (1, 128), device)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(b_data.clone(), (128, 64), device)?.to_dtype(DType::BF16)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = c.to_vec2::<f32>()?.into_iter().flatten().collect();
    // Verify against CPU reference
    let a_cpu = Tensor::from_vec(a_data, (1, 128), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let b_cpu = Tensor::from_vec(b_data, (128, 64), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let c_cpu = a_cpu.matmul(&b_cpu)?.to_dtype(DType::F32)?;
    let expected: Vec<f32> = c_cpu.to_vec2::<f32>()?.into_iter().flatten().collect();
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        let tol = e.abs() * 0.05 + 0.1;
        assert!((e - g).abs() < tol, "gemv_large[{i}]: expected {e}, got {g}");
    }

    // Test 3: Batched GEMV — [2, 1, 4] @ [2, 4, 3] = [2, 1, 3]
    let a_batch = Tensor::from_vec(
        vec![1.0f32, 2.0, 3.0, 4.0,
             5.0, 6.0, 7.0, 8.0],
        (2, 1, 4),
        device,
    )?.to_dtype(DType::BF16)?;
    let b_batch = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0,
             0.0, 0.0, 0.0,
             // batch 2
             2.0, 0.0, 0.0,
             0.0, 2.0, 0.0,
             0.0, 0.0, 2.0,
             0.0, 0.0, 0.0],
        (2, 4, 3),
        device,
    )?.to_dtype(DType::BF16)?;
    let c_batch = a_batch.matmul(&b_batch)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = c_batch.to_vec3::<f32>()?.into_iter().flatten().flatten().collect();
    // batch 0: [1,2,3,4] @ [[1,0,0],[0,1,0],[0,0,1],[0,0,0]] = [1,2,3]
    // batch 1: [5,6,7,8] @ [[2,0,0],[0,2,0],[0,0,2],[0,0,0]] = [10,12,14]
    let expected = vec![1.0, 2.0, 3.0, 10.0, 12.0, 14.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.5, "gemv_batch[{i}]: expected {e}, got {g}");
    }

    // Test 4: K=6 (even but not divisible by 4) — exercises GEMV remainder path
    let a = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 6), device)?.to_dtype(DType::BF16)?;
    let b = Tensor::from_vec(
        vec![1.0f32, 0.0, 0.0,
             0.0, 1.0, 0.0,
             0.0, 0.0, 1.0,
             1.0, 1.0, 1.0,
             2.0, 2.0, 2.0,
             0.5, 0.5, 0.5],
        (6, 3),
        device,
    )?.to_dtype(DType::BF16)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = c.to_vec2::<f32>()?.into_iter().flatten().collect();
    // [1,2,3,4,5,6] @ [[1,0,0],[0,1,0],[0,0,1],[1,1,1],[2,2,2],[0.5,0.5,0.5]]
    // = [1+4+10+3, 2+4+10+3, 3+4+10+3] = [18, 19, 20]
    let expected = vec![18.0, 19.0, 20.0];
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        assert!((e - g).abs() < 0.5, "gemv_k6[{i}]: expected {e}, got {g}");
    }

    // Test 5: K=8 (divisible by 4, exactly 2 vec4 chunks, no remainder)
    let a = Tensor::from_vec(vec![1.0f32; 8], (1, 8), device)?.to_dtype(DType::BF16)?;
    let b_data: Vec<f32> = (0..8*4).map(|i| (i as f32) * 0.1).collect();
    let b = Tensor::from_vec(b_data.clone(), (8, 4), device)?.to_dtype(DType::BF16)?;
    let c = a.matmul(&b)?.to_dtype(DType::F32)?;
    let result: Vec<f32> = c.to_vec2::<f32>()?.into_iter().flatten().collect();
    let a_cpu = Tensor::from_vec(vec![1.0f32; 8], (1, 8), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let b_cpu = Tensor::from_vec(b_data, (8, 4), &Device::Cpu)?.to_dtype(DType::BF16)?;
    let c_cpu = a_cpu.matmul(&b_cpu)?.to_dtype(DType::F32)?;
    let expected: Vec<f32> = c_cpu.to_vec2::<f32>()?.into_iter().flatten().collect();
    for (i, (e, g)) in expected.iter().zip(result.iter()).enumerate() {
        let tol = e.abs() * 0.05 + 0.1;
        assert!((e - g).abs() < tol, "gemv_k8[{i}]: expected {e}, got {g}");
    }

    eprintln!("[OK] gemv BF16 (simple, large K=128, batched, K=6 remainder, K=8 exact)");
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
        ("linear_matmul", Box::new(test_linear_matmul)),
        ("gemv_bf16", Box::new(test_gemv_bf16)),
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
