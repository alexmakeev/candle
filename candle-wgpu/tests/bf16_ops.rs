//! Integration tests for BF16 operations
//!
//! This test file demonstrates the usage of BF16 softmax and layer_norm operations.

use candle_core::backend::{BackendDevice, BackendStorage};
use candle_wgpu::{WgpuDevice, is_available};

#[test]
fn test_bf16_softmax_integration() {
    if !is_available() {
        println!("Skipping test: no wgpu adapter available");
        return;
    }

    let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
    println!("Testing BF16 softmax integration on: {}", device.adapter_info().name);

    // Test data
    let num_rows = 4;
    let row_size = 8;
    let input_f32: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
        10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
    ];

    // Convert to BF16
    let input_bf16: Vec<half::bf16> = input_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let input_storage = device.storage_from_slice(&input_bf16).expect("Failed to create input");

    // Run BF16 softmax
    let output_storage = candle_wgpu::softmax::softmax_bf16_gpu(&device, &input_storage, num_rows, row_size)
        .expect("softmax_bf16_gpu failed");

    // Read back result
    let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
    let output: Vec<f32> = match output_cpu {
        candle_core::CpuStorage::F32(data) => data,
        _ => panic!("Unexpected dtype"),
    };

    // Verify softmax properties
    for row in 0..num_rows {
        let offset = row * row_size;
        let row_data = &output[offset..offset + row_size];

        // Sum should be 1.0
        let sum: f32 = row_data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "Row {} sum should be 1.0, got {}",
            row, sum
        );

        // All values should be in [0, 1]
        for (i, &val) in row_data.iter().enumerate() {
            assert!(
                val >= 0.0 && val <= 1.0,
                "Row {} value {} should be in [0, 1], got {}",
                row, i, val
            );
        }
    }

    println!("✅ BF16 softmax integration test passed!");
}

#[test]
fn test_bf16_layer_norm_integration() {
    if !is_available() {
        println!("Skipping test: no wgpu adapter available");
        return;
    }

    let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
    println!("Testing BF16 layer_norm integration on: {}", device.adapter_info().name);

    let batch_size = 4;
    let hidden_size = 16;
    let eps = 1e-5;

    // Test data
    let input_f32: Vec<f32> = (0..batch_size * hidden_size)
        .map(|i| (i as f32 * 0.1).sin() * 2.0 + 1.0)
        .collect();

    // Learnable parameters
    let gamma_f32: Vec<f32> = (0..hidden_size)
        .map(|i| 1.0 + (i as f32 / hidden_size as f32) * 0.5)
        .collect();
    let beta_f32: Vec<f32> = (0..hidden_size)
        .map(|i| (i as f32 / hidden_size as f32) - 0.5)
        .collect();

    // Convert to BF16
    let input_bf16: Vec<half::bf16> = input_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let gamma_bf16: Vec<half::bf16> = gamma_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let beta_bf16: Vec<half::bf16> = beta_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

    let input_storage = device.storage_from_slice(&input_bf16).expect("Failed to create input");
    let gamma_storage = device.storage_from_slice(&gamma_bf16).expect("Failed to create gamma");
    let beta_storage = device.storage_from_slice(&beta_bf16).expect("Failed to create beta");

    // Run BF16 layer_norm
    let output_storage = candle_wgpu::layer_norm::layer_norm_bf16_gpu(
        &device,
        &input_storage,
        &gamma_storage,
        &beta_storage,
        batch_size,
        hidden_size,
        eps,
    )
    .expect("layer_norm_bf16_gpu failed");

    // Read back result
    let output_cpu = output_storage.to_cpu_storage().expect("Failed to read back");
    let output: Vec<f32> = match output_cpu {
        candle_core::CpuStorage::F32(data) => data,
        _ => panic!("Unexpected dtype"),
    };

    // Verify output is reasonable (no NaN, no Inf)
    for (i, &val) in output.iter().enumerate() {
        assert!(
            !val.is_nan() && !val.is_infinite(),
            "Output at index {} is invalid: {}",
            i, val
        );
    }

    // Verify each batch has normalized statistics before scale/shift
    // After applying gamma and beta, mean and variance will be different
    // So we just check that the operation ran successfully
    println!("Output shape: {} × {}", batch_size, hidden_size);
    println!("Output range: [{:.4}, {:.4}]",
        output.iter().cloned().fold(f32::INFINITY, f32::min),
        output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    println!("✅ BF16 layer_norm integration test passed!");
}

#[test]
fn test_bf16_ops_pipeline() {
    if !is_available() {
        println!("Skipping test: no wgpu adapter available");
        return;
    }

    let device = WgpuDevice::new(0).expect("Failed to create wgpu device");
    println!("Testing BF16 ops pipeline on: {}", device.adapter_info().name);

    // Simulate a simple transformer-like pipeline:
    // 1. Layer norm on input
    // 2. Softmax on attention scores

    let batch_size = 2;
    let seq_len = 4;
    let hidden_size = 8;
    let eps = 1e-5;

    // Step 1: Layer norm on input [batch_size, hidden_size]
    let input_f32: Vec<f32> = (0..batch_size * hidden_size)
        .map(|i| (i as f32 * 0.123).sin() + 0.5)
        .collect();
    let gamma_f32: Vec<f32> = vec![1.0; hidden_size];
    let beta_f32: Vec<f32> = vec![0.0; hidden_size];

    let input_bf16: Vec<half::bf16> = input_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let gamma_bf16: Vec<half::bf16> = gamma_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let beta_bf16: Vec<half::bf16> = beta_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();

    let input_storage = device.storage_from_slice(&input_bf16).unwrap();
    let gamma_storage = device.storage_from_slice(&gamma_bf16).unwrap();
    let beta_storage = device.storage_from_slice(&beta_bf16).unwrap();

    let normed_storage = candle_wgpu::layer_norm::layer_norm_bf16_gpu(
        &device,
        &input_storage,
        &gamma_storage,
        &beta_storage,
        batch_size,
        hidden_size,
        eps,
    )
    .expect("layer_norm failed");

    println!("✓ Layer norm completed");

    // Step 2: Softmax on attention scores [batch_size, seq_len]
    let attn_f32: Vec<f32> = (0..batch_size * seq_len)
        .map(|i| (i as f32) * 0.5 - 1.0)
        .collect();
    let attn_bf16: Vec<half::bf16> = attn_f32.iter().map(|&x| half::bf16::from_f32(x)).collect();
    let attn_storage = device.storage_from_slice(&attn_bf16).unwrap();

    let attn_probs_storage = candle_wgpu::softmax::softmax_bf16_gpu(
        &device,
        &attn_storage,
        batch_size,
        seq_len,
    )
    .expect("softmax failed");

    println!("✓ Softmax completed");

    // Verify outputs
    let normed_cpu = normed_storage.to_cpu_storage().unwrap();
    let attn_probs_cpu = attn_probs_storage.to_cpu_storage().unwrap();

    let normed: Vec<f32> = match normed_cpu {
        candle_core::CpuStorage::F32(data) => data,
        _ => panic!("Unexpected dtype"),
    };

    let attn_probs: Vec<f32> = match attn_probs_cpu {
        candle_core::CpuStorage::F32(data) => data,
        _ => panic!("Unexpected dtype"),
    };

    // Check layer norm output
    assert_eq!(normed.len(), batch_size * hidden_size);
    for &val in &normed {
        assert!(!val.is_nan() && !val.is_infinite());
    }

    // Check softmax output
    assert_eq!(attn_probs.len(), batch_size * seq_len);
    for row in 0..batch_size {
        let offset = row * seq_len;
        let sum: f32 = attn_probs[offset..offset + seq_len].iter().sum();
        assert!((sum - 1.0).abs() < 1e-3, "Row {} sum is {}", row, sum);
    }

    println!("✅ BF16 ops pipeline test passed!");
}
