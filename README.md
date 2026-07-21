<div align="center">

# MiniGPT
### A GPT-Style Language Model Built Completely from Scratch using NumPy

<p>
A complete implementation of a <b>decoder-only Transformer</b> with <b>manual forward propagation, manual backpropagation, and Adam optimization</b>, built entirely using <b>Python + NumPy</b> without relying on PyTorch, TensorFlow, or automatic differentiation.
</p>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)
![Transformer](https://img.shields.io/badge/Architecture-Decoder--Only%20Transformer-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

# Overview

Modern Large Language Models are built upon the Transformer architecture.

Instead of using high-level deep learning frameworks, this project focuses on understanding **how a GPT model actually works internally** by implementing every important component manually.

Every major building block—including embeddings, layer normalization, causal self-attention, feed-forward networks, gradient computation, and optimization—has been implemented from scratch using only **NumPy**.

The model is trained on the **Tiny Shakespeare** dataset and learns to generate Shakespeare-style text autoregressively.

---

# Features

- Character-level Tokenizer
- Token Embeddings
- Learnable Positional Embeddings
- Decoder-Only Transformer Architecture
- Causal Self-Attention
- Layer Normalization
- GELU Activation
- Feed Forward Network
- Residual Connections
- Manual Forward Propagation
- Manual Backpropagation
- Cross Entropy Loss
- Adam Optimizer
- Temperature-Based Text Generation
- Checkpoint Saving & Loading
- Autoregressive Text Generation

---

# What Makes This Project Different?

Many educational GPT implementations rely on

```python
loss.backward()
optimizer.step()
```

This project does **not**.

Instead, gradients are manually derived and implemented for every trainable layer.

Every parameter update is computed explicitly using NumPy.

No automatic differentiation framework is used during training.

---

# Architecture

```
                 Input Text
                      │
                      ▼
          Character Tokenizer
                      │
                      ▼
     Token + Positional Embeddings
                      │
                      ▼
          ┌─────────────────────┐
          │ Transformer Block 1 │
          └─────────────────────┘
                      │
                      ▼
          ┌─────────────────────┐
          │ Transformer Block 2 │
          └─────────────────────┘
                      │
                      ▼
              Final LayerNorm
                      │
                      ▼
             Linear Projection
                      │
                      ▼
                  Softmax
                      │
                      ▼
          Next Character Prediction
```

---

# Components Implemented

| Component | Status |
|-----------|:------:|
| Character Tokenizer | ✅ |
| Token Embeddings | ✅ |
| Positional Embeddings | ✅ |
| LayerNorm (Forward) | ✅ |
| LayerNorm (Backward) | ✅ |
| Causal Self-Attention | ✅ |
| Self-Attention Backpropagation | ✅ |
| Feed Forward Network | ✅ |
| GELU Activation | ✅ |
| GELU Derivative | ✅ |
| Residual Connections | ✅ |
| Decoder Transformer Blocks | ✅ |
| Cross Entropy Loss | ✅ |
| Adam Optimizer | ✅ |
| Checkpoint Saving | ✅ |
| Checkpoint Loading | ✅ |
| Temperature Sampling | ✅ |
| Text Generation | ✅ |

---

# Project Structure

```
MiniGPT/

│── mini_gpt.py
│── generate.py
│── tinyshakespeare.txt
│── generated_shakespeare.txt
│── model_weights.npz
│── tokenizer.npz
│── requirements.txt
│── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/MiniGPT.git
```

Move into the project directory

```bash
cd MiniGPT
```

Install NumPy

```bash
pip install numpy
```

or

```bash
pip install -r requirements.txt
```

---

# Dataset

This project uses the **Tiny Shakespeare** dataset.

If the dataset is missing, download it from

https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

and place it inside the project directory.

---

# Training the Model

Run

```bash
python mini_gpt.py
```

Example output

```text
============================================================
MINI GPT — Full Backprop Through Every Layer
============================================================

Dataset: 1,115,394 characters

Vocabulary: 65 unique characters

Step 0     | Loss: 4.17
Step 100   | Loss: 3.15
Step 500   | Loss: 2.71
Step 1000  | Loss: 2.56
Step 2000  | Loss: 2.32
Step 2900  | Loss: 2.56
```

After training, the model automatically saves

```
model_weights.npz
tokenizer.npz
```

---

# Generating Text

Generate Shakespeare-like text using the trained model

```bash
python generate.py
```

Example output

```text
ROMEO:

ANANUGLICEDO:

Thor thofeak thor bere utharen b:
Be marg...
```

The generated text is intentionally imperfect because the model is a small character-level Transformer trained on consumer hardware.

---

# Training Pipeline

```
Dataset

↓

Tokenizer

↓

Embedding

↓

Transformer

↓

Cross Entropy Loss

↓

Manual Backpropagation

↓

Adam Optimizer

↓

Updated Weights

↓

Checkpoint Saved
```

---

# Learning Objectives

The primary goal of this project was to deeply understand the mathematics and implementation behind Transformer-based language models.

Topics explored include:

- Embedding Layers
- Layer Normalization
- Attention Mechanism
- Residual Learning
- Gradient Flow
- Manual Backpropagation
- Adam Optimization
- Language Modeling
- Sequence Prediction
- Autoregressive Generation

---

# Future Improvements

- Multi-Head Attention
- Byte Pair Encoding (BPE)
- Rotary Position Embeddings (RoPE)
- RMSNorm
- SwiGLU
- KV Cache
- Top-k Sampling
- Top-p Sampling
- Flash Attention
- Mixed Precision Training
- GPU Acceleration

---

# References

**Attention Is All You Need**

Ashish Vaswani et al.

https://arxiv.org/abs/1706.03762

---

**Improving Language Understanding by Generative Pre-Training**

Alec Radford et al.

https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf

---

# Acknowledgements

This project was built as a first-principles implementation to better understand the inner workings of GPT-style language models by manually implementing the core algorithms rather than relying on deep learning frameworks.

---

# License

This project is licensed under the MIT License.

---

<div align="center">

⭐ If you found this project interesting, consider giving it a star!

</div>