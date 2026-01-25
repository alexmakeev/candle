#!/usr/bin/env python3
"""Analyze raw codec tokens from Talker before offset"""
import json
import sys

tokens_file = sys.argv[1]
with open(tokens_file, 'r') as f:
    flat_tokens = json.load(f)

# Reverse the offset: token = flat_token - (cb * 4096)
# Each position has 8 codebooks
num_codebooks = 8
codebook_size = 4096

raw_tokens = []
for i, flat in enumerate(flat_tokens):
    cb = i % num_codebooks
    raw = flat - cb * codebook_size
    raw_tokens.append(raw)

print(f"Raw tokens (before offset):")
print(f"  Total: {len(raw_tokens)}")
print(f"  Min: {min(raw_tokens)}")
print(f"  Max: {max(raw_tokens)}")
print(f"  Unique: {len(set(raw_tokens))}")

# Group by codebook
for cb in range(num_codebooks):
    cb_tokens = [raw_tokens[i] for i in range(cb, len(raw_tokens), num_codebooks)]
    print(f"  Codebook {cb}: min={min(cb_tokens)}, max={max(cb_tokens)}, unique={len(set(cb_tokens))}")

# Check if all tokens are in valid range [0, 4096)
invalid = [t for t in raw_tokens if t < 0 or t >= codebook_size]
if invalid:
    print(f"\n⚠️  WARNING: {len(invalid)} tokens out of range!")
    print(f"  Invalid tokens: {invalid[:10]}")
