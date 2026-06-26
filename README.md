# Decoder Transformer From Scratch (PyTorch)

A from-scratch implementation of a decoder-only Transformer (GPT) in PyTorch built while studying the architecture described in *Attention Is All You Need* *Layer Normalization*  *Attention Scheme Inspired Softmax Regressio*.

The goal of this project is not simply to train a language model, but to understand every component of the Transformer by implementing it manually.

---

## Overview

This project progressively builds a GPT language model from first principles:

1. Bigram Language Model
2. Token & Positional Embeddings
3. Self-Attention
4. Multi-Head Attention
5. Feed Forward Networks
6. Residual Connections
7. Layer Normalization (Pre-Norm)
8. Transformer Blocks
9. Character-level Text Generation

Every component is implemented manually using PyTorch without Transformer libraries.

---

## Architecture

```
Input Tokens
      │
      ▼
Token Embedding
      │
      ▼
Positional Embedding
      │
      ▼
N × Transformer Blocks

    ┌─────────────────────────────┐
    │ LayerNorm                   │
    │ Multi-Head Self Attention   │
    │ Residual Connection          │
    │ LayerNorm                   │
    │ Feed Forward Network         │
    │ Residual Connection          │
    └─────────────────────────────┘

      │
      ▼
Linear Projection
      │
      ▼
Softmax
      │
      ▼
Next Token Prediction
```

---

## Concepts Implemented

### Language Modeling

- Character-level tokenization
- Next-token prediction
- Cross-entropy loss
- Autoregressive generation

### Embeddings

- Token embeddings
- Learned positional embeddings

### Self Attention

- Query / Key / Value projections
- Scaled dot-product attention
- Causal masking
- Softmax attention weights

### Multi-Head Attention

- Parallel attention heads
- Head concatenation
- Output projection

### Transformer Block

- Pre-Norm Layer Normalization
- Residual Connections
- Feed Forward Network
- Stacked decoder blocks

### Training

- AdamW optimizer
- Batch sampling
- Train / Validation split
- Loss estimation during training

---

## Repository Structure

```
.
├── bigram.py          # Bigram language model
├── v2.py              # Decoder-only GPT implementation
├── input.txt          # Training corpus
└── README.md
```

---

## Example Training

```
step 0:
train loss 4.18
val loss 4.22

step 500:
train loss 2.63
val loss 2.70

step 1000:
train loss 2.31
val loss 2.40
```

---

## Example Generation

```
ROMEO:
Good morrow, my lord...
```

(Generated after training on Tiny Shakespeare.)

---

## What I Learned

Through implementing the model from scratch I gained a practical understanding of:

- Why embeddings are learned
- Why positional information is required
- How self-attention computes token interactions
- Why causal masking prevents future token leakage
- How multi-head attention learns multiple relationships
- Why feed-forward networks expand to 4× hidden size
- Why residual connections stabilize deep networks
- Why modern GPT models use Pre-LayerNorm
- How autoregressive generation works token-by-token

---

## Future Improvements

- Rotary Positional Embeddings (RoPE)
- Flash Attention
- KV Cache for faster inference
- Mixed Precision (AMP)
- Weight tying
- GPT-2 tokenizer (BPE)
- Text sampling (Top-k / Top-p / Temperature)
- Model checkpointing
- TensorBoard / Weights & Biases logging
- HuggingFace compatible inference

---
