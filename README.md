# Transformer-Based Bug Triage via Dual-Head GPT-2 Fine-Tuning

A large language model fine-tuned to classify software bug reports by category and severity, built on a custom GPT-2 implementation trained from scratch in PyTorch.

# Overview
This project fine-tunes a GPT-2 (124M parameter) language model on 50,000 bug reports to simultaneously predict two properties of a commit message the bug category (16 classes) and its severity level (4 classes) using a dual-head classification architecture. The entire transformer backbone was implemented from scratch in PyTorch without HuggingFace or other high-level abstractions, and pre-trained GPT-2 weights were loaded manually from OpenAI's public checkpoint.

# Key Technical Highlights

1. Custom transformer implementation — multi-head self-attention, positional embeddings, layer normalization, and feed-forward blocks built from scratch in PyTorch
2. Manual weight loading — GPT-2 pre-trained weights transferred directly from TensorFlow checkpoint into the custom PyTorch architecture
3. Dual-head fine-tuning — two classification heads trained simultaneously on the same frozen backbone, predicting bug category and severity in a single forward pass
4. Selective layer unfreezing — 117M backbone parameters frozen; only the final transformer block, layer norm, and classification heads (~7M parameters) are trained
5. End-to-end pipeline — data preprocessing, class balancing, BPE tokenization, training loop, evaluation, and model serialization
Interactive demo — Gradio web UI for live inference with confidence scores and full probability distributions across all classes

## Project Structure
```
├── model.py          # Custom transformer + dual-head classifier
├── gpt_weights.py    # GPT-2 weight download and loading utilities
├── dataset.py        # Dataset, tokenization, class balancing
├── train.py          # Training loop, loss, accuracy
├── inference.py      # Single-input inference function
├── main.py           # End-to-end CLI entry point
├── app.py            # Gradio web UI
└── demo.html         # Standalone browser demo
```

# Quickstart

```bash
pip install -r requirements.txt
python main.py --data bug_dataset_50k.csv --epochs 5
python app.py   # launch Gradio UI at localhost:7860
```

# Tech Stack
Python · PyTorch · GPT-2 · Transfer Learning · tiktoken · Gradio · Pandas
