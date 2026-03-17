import torch
import json
import torch.nn.functional as F
from model import LanguageModel, block_size

# load hyperparameters
batch_size = 32
max_iters = 2000 # Increased further for better convergence
eval_interval = 100
learning_rate = 1e-4 # Lowered for more stability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the vocabulary created during pre-training
with open('vocab.json', 'r') as f:
    vocab = json.load(f)
stoi = vocab['stoi']
itos = {int(k): v for k, v in vocab['itos'].items()}
vocab_size = len(stoi)

encode = lambda s: [stoi.get(c, stoi[' ']) for c in s]

# Load the QA dataset from file
try:
    with open('qa_dataset.json', 'r') as f:
        qa_dataset = json.load(f)
    print(f"Loaded {len(qa_dataset)} QA pairs for fine-tuning.")
except FileNotFoundError:
    print("Warning: qa_dataset.json not found. Generating default data.")
    qa_dataset = [
        "Q: What is a foundation model?\nA: A model trained on massive datasets to learn language.\n",
    ] * 20

# Encode the QA dataset into a continuous stream for fine-tuning
qa_text = "".join(qa_dataset)
data = torch.tensor(encode(qa_text), dtype=torch.long)

def get_batch():
    # generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    losses = torch.zeros(10)
    for k in range(10):
        X, Y = get_batch()
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()
    return out

# Initialize model and load pre-trained weights
model = LanguageModel(vocab_size=vocab_size)
print("Loading pre-trained weights...")
try:
    model.load_state_dict(torch.load('pretrained_model.pt', map_location=device))
    print("Pre-trained weights loaded successfully.")
except FileNotFoundError:
    print("Warning: pre-trained weights not found. You should run train.py first!")
    
model = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Starting Instruction Fine-tuning for {max_iters} iterations on {device}...")
for iter in range(max_iters):
    
    # evaluate the loss periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        loss_val = estimate_loss()
        print(f"step {iter}: fine-tuning loss {loss_val:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch()
    
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Fine-tuning completed.")

# Save the perfectly fine-tuned model weights
torch.save(model.state_dict(), 'finetuned_model.pt')
print("Model saved to finetuned_model.pt")
