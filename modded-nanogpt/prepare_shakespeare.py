#!/usr/bin/env python3
"""
Prepare Shakespeare dataset for rapid nGPT experimentation.
Downloads tiny Shakespeare (~1MB) and converts to binary format.
"""

import os
import pickle
import urllib.request
import numpy as np
import tiktoken

# Download Shakespeare dataset
print("Downloading Shakespeare dataset...")
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
with urllib.request.urlopen(url) as response:
    data = response.read().decode('utf-8')

print(f"Downloaded {len(data):,} characters")

# Tokenize with GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)
print(f"Tokenized to {len(tokens):,} tokens")

# Split into train/val
n = len(tokens)
train_tokens = tokens[:int(n*0.9)]
val_tokens = tokens[int(n*0.9):]

print(f"Train: {len(train_tokens):,} tokens")
print(f"Val: {len(val_tokens):,} tokens")

# Save as binary files (matching FineWeb format)
os.makedirs('data/shakespeare', exist_ok=True)

# Write train data
with open('data/shakespeare/shakespeare_train_000001.bin', 'wb') as f:
    # Header: magic number, version, number of tokens
    header = np.array([20240520, 1, len(train_tokens)], dtype=np.int32)
    header.tofile(f)
    # Data: tokens as uint16
    tokens_array = np.array(train_tokens, dtype=np.uint16)
    tokens_array.tofile(f)

# Write val data
with open('data/shakespeare/shakespeare_val_000000.bin', 'wb') as f:
    header = np.array([20240520, 1, len(val_tokens)], dtype=np.int32)
    header.tofile(f)
    tokens_array = np.array(val_tokens, dtype=np.uint16)
    tokens_array.tofile(f)

print(f"\n✓ Shakespeare dataset prepared:")
print(f"  Train: data/shakespeare/shakespeare_train_000001.bin")
print(f"  Val: data/shakespeare/shakespeare_val_000000.bin")
print(f"  Total size: ~{(len(train_tokens) + len(val_tokens)) * 2 / 1024:.1f} KB")
