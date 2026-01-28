#!/bin/bash
# Test script to verify qwen3_omni_wgpu example compiles

set -e

echo "Testing qwen3_omni_wgpu example compilation..."
echo "==============================================="

# Check if we're in the right directory
if [ ! -f "candle-examples/examples/qwen3_omni_wgpu/main.rs" ]; then
    echo "Error: Must run from candle-16b root directory"
    exit 1
fi

# Try to build the example
echo "Running: cargo build --example qwen3_omni_wgpu"
if cargo build --example qwen3_omni_wgpu 2>&1 | tee /tmp/build.log; then
    echo ""
    echo "✓ Example compiled successfully!"
    echo ""
    echo "You can now run it with:"
    echo "  cargo run --example qwen3_omni_wgpu -- \\"
    echo "    --weight-path /home/lluda/.cache/huggingface/hub/models--Qwen--Qwen3-Omni \\"
    echo "    --prompt 'What is deep learning?'"
else
    echo ""
    echo "✗ Build failed. Check errors above."
    exit 1
fi
