import torch
import torch.nn as nn
from model import LanguageModel, block_size
import os
import json

# hyperparameters
batch_size = 32
max_iters = 500 # Keep it small for quick training in this env
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 50

# 1. Prepare data
# We'll use a simple dummy text dataset representing a miniature "Wikipedia" or "Shakespeare"
text = """
The foundation of a large language model starts with data. The model learns to predict the next token based on the previous tokens in a sequence. By repeating this process over and over across massive datasets, the model learns grammar, facts, and reasoning abilities. This process is called pre-training.

After pre-training, the model is a "base model" that is good at continuing text, but not necessarily good at answering questions or following instructions. To fix this, we use a process called fine-tuning. Fine-tuning aligns the model's outputs to human expectations, making it a helpful AI assistant.

This toy example demonstrates both pre-training and fine-tuning using a custom Transformer architecture built from scratch in PyTorch. The Transformer uses self-attention mechanisms to weigh the importance of different words in a sequence, allowing it to capture complex relationships and context.
"""

# add more text to make it slightly larger
text = text * 50

# Ensure all characters needed for Q&A are in the vocabulary
text += "Q:Whatisafoundationmodel?W"

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocab size: {vocab_size}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Save the vocab so we can use it during fine-tuning and inference
with open('vocab.json', 'w') as f:
    json.dump({'stoi': stoi, 'itos': itos}, f)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Initialize model
model = LanguageModel(vocab_size=vocab_size)
model = model.to(device)
print(f"Model instantiated with {sum(p.numel() for p in model.parameters())} parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Starting Pre-training for {max_iters} iterations on {device}...")
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Pre-training completed.")

# Save the pre-trained model weights
torch.save(model.state_dict(), 'pretrained_model.pt')
print("Model saved to pretrained_model.pt")
