# Custom Foundational LLM

A character-level Transformer language model built from scratch in PyTorch, featuring a full pipeline for pre-training and instruction fine-tuning.

## 🚀 Overview

This project demonstrates the lifecycle of a Large Language Model (LLM) on a miniature scale. It includes:
- **Custom Transformer Architecture**: Multi-head self-attention, feed-forward networks, and residual connections.
- **Pre-training Pipeline**: Learning language patterns from raw text.
- **Instruction Fine-tuning**: Aligning the base model to follow a `Q: A:` format.
- **Large Synthetic Dataset**: A script to generate 5,000+ QA pairs for robust training.
- **Interactive Chat**: A CLI interface to interact with the fine-tuned model.

## 🛠️ Architecture

The model is defined in `model.py` and uses:
- **Embedding Dimension**: 64
- **Attention Heads**: 4
- **Layers**: 4
- **Context Window (Block Size)**: 128 characters
- **Parameters**: ~212,000

## 📂 Project Structure

- `model.py`: The core Transformer architecture.
- `train.py`: Pre-trains the model on a sample text dataset (generates `pretrained_model.pt`).
- `generate_data.py`: Generates a large synthetic dataset of 5,000 QA pairs (`qa_dataset.json`).
- `finetune.py`: Loads pre-trained weights and fine-tunes on the QA dataset (generates `finetuned_model.pt`).
- `chat.py`: The interactive chat interface.

## 🚦 Getting Started

### 1. Install Dependencies
```bash
pip install torch
```

### 2. Prepare Data
Generate the 5,000-sample synthetic dataset:
```bash
python generate_data.py
```

### 3. Training Pipeline
Run the full training cycle:
```bash
# Pre-train the base model
python train.py

# Fine-tune for instructions
python finetune.py
```

### 4. Chat with the AI
Start an interactive session:
```bash
python chat.py
```

## 📝 Usage Notes
- **Hardware**: The model runs efficiently on CPU but will use CUDA if available.
- **Coherence**: This is a character-level model with a small parameter count (~212k). While it follows instruction formats well, its "knowledge" is limited to the provided datasets.
- **Format**: Always ask questions using natural language. The system automatically formats them for the model.
