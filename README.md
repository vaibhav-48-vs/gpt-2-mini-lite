# Mini GPT-2

<p align="center">
  <img src="data/mini-gpt-banner.svg" alt="Mini GPT-2 banner" width="100%" />
</p>

A compact GPT-style language model implemented in PyTorch from scratch. This project trains a decoder-only transformer on text data, uses Rotary Positional Embeddings (RoPE), Multi-Query Attention (MQA), and a lightweight training loop inspired by modern LLM architectures.

## Overview

This repo contains a minimal but practical implementation of a GPT-like model for learning language modeling fundamentals. The main script focuses on:

- Decoder-only transformer architecture
- Causal masked attention
- Rotary positional encoding
- Multi-query attention optimization
- Training with AdamW and cosine learning rate decay
- Mixed precision support for CUDA-enabled GPUs
- Text generation from a trained checkpoint

The current code is built around the file `gpt_2 _mini.py` and uses the `shakespeare.txt` corpus for training.

## Project Structure

```text
.
├── README.md
├── gpt_2 _mini.py
├── gpt.py
├── shakespeare.txt
├── assets/
│   └── mini-gpt-banner.svg
└── ckpt.pt   # generated after training
```

## Features

- GPT-2-inspired transformer block design
- Efficient multi-head attention with shared K/V projection
- RoPE-based positional encoding for better sequence modeling
- Layer normalization and feed-forward network blocks
- Training pipeline with batching and tokenization
- CUDA optimization and autocast support
- Text generation using sampling-based decoding

## Requirements

Install the following dependencies:

```bash
pip install torch tiktoken numpy
```

Optional:

- CUDA-capable GPU for faster training
- NVIDIA drivers and PyTorch GPU build

## How It Works

The model is a compact transformer decoder that:

1. Tokenizes the text using the GPT-2 tokenizer
2. Converts tokens into embeddings
3. Processes them through a stack of transformer blocks
4. Predicts the next token for each position
5. Trains using cross-entropy loss and backpropagation

The training objective is standard autoregressive language modeling:

$$
\mathcal{L} = -\sum_t \log P(x_{t+1} \mid x_{\le t})
$$

## Running the Project

Run the training script:

```bash
python "gpt_2 _mini.py"
```

If the file path is a problem on Linux/macOS, make sure to quote it because of the space in the filename.

## Training Configuration

The script includes settings such as:

- Batch size
- Block size
- Learning rate
- Warmup schedule
- Gradient clipping
- Evaluation interval
- Training steps

You can tune the hyperparameters near the configuration section of the script to match your hardware and dataset size.

## Dataset

The project uses `shakespeare.txt` as the training corpus. This small text dataset is ideal for experimenting with transformer training and quick iterations.

## Model Notes

This is a learning-focused implementation rather than a production-scale LLM. It is intentionally lightweight and readable so it can be used as a study project or educational reference.

Helpful technical notes from the implementation:

- Weight tying is used between embedding and output projection
- Mixed precision is enabled automatically on CUDA devices
- Training is optimized for fast iteration on small datasets
- The model is designed to be easy to understand and modify

## Example Usage

After training, the script saves the model weights to:

```bash
ckpt.pt
```

You can modify the generation logic in the script to sample text from the model after training.

## Recommended Next Steps

- Increase dataset size and training steps
- Add validation loss tracking
- Save prompts and generated samples to a results folder
- Experiment with different model sizes and learning rates
- Add checkpoint loading and inference utilities

## License

This project is intended for educational and research purposes. Add a license if you plan to publish it publicly on GitHub.

## Acknowledgements

This project draws inspiration from GPT-style decoder-only transformer architectures and modern LLM training practices.

---

Built for learning, experimentation, and small-scale text generation.

