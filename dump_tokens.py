#!/usr/bin/env python3
"""Analyze Talker codec tokens saved as binary file.

Format: flat u32 array [seq_len * num_codebooks]
"""
import sys
import numpy as np

def analyze_tokens(filepath, num_codebooks=8):
    """Load and analyze codec tokens."""
    
    # Load binary u32 data
    tokens = np.fromfile(filepath, dtype=np.uint32)
    
    print(f"Total tokens: {len(tokens)}")
    print(f"Codebooks: {num_codebooks}")
    
    if len(tokens) % num_codebooks != 0:
        print(f"WARNING: token count ({len(tokens)}) not divisible by {num_codebooks}")
        return
    
    seq_len = len(tokens) // num_codebooks
    print(f"Sequence length: {seq_len}")
    
    # Reshape to [seq_len, num_codebooks]
    tokens_2d = tokens.reshape(seq_len, num_codebooks)
    
    print(f"\nToken statistics:")
    print(f"  Min: {tokens.min()}")
    print(f"  Max: {tokens.max()}")
    print(f"  Mean: {tokens.mean():.1f}")
    print(f"  Std: {tokens.std():.1f}")
    
    # Check for invalid tokens (>= 32768)
    invalid = np.sum(tokens >= 32768)
    if invalid > 0:
        print(f"  WARNING: {invalid} tokens >= 32768 (out of range!)")
    
    # Per-codebook statistics
    print(f"\nPer-codebook statistics:")
    for cb in range(num_codebooks):
        cb_tokens = tokens_2d[:, cb]
        print(f"  Codebook {cb}: min={cb_tokens.min()}, max={cb_tokens.max()}, mean={cb_tokens.mean():.1f}")
    
    # Check for repeated patterns
    print(f"\nUnique tokens per codebook:")
    for cb in range(num_codebooks):
        cb_tokens = tokens_2d[:, cb]
        unique = len(np.unique(cb_tokens))
        print(f"  Codebook {cb}: {unique} unique values (out of {seq_len})")
    
    # Check if all same
    all_same = []
    for cb in range(num_codebooks):
        cb_tokens = tokens_2d[:, cb]
        if len(np.unique(cb_tokens)) == 1:
            all_same.append(cb)
    
    if all_same:
        print(f"\n  WARNING: Codebooks {all_same} have all identical tokens!")
    
    # Show first few tokens
    print(f"\nFirst 5 positions:")
    for i in range(min(5, seq_len)):
        row = tokens_2d[i]
        print(f"  Pos {i}: {row}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 dump_tokens.py tokens.bin [num_codebooks]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    num_codebooks = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    
    analyze_tokens(filepath, num_codebooks)
