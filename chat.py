import torch
import json
import sys
from model import LanguageModel

# load configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the vocabulary to translate back to text
print("Loading vocabulary...")
try:
    with open('vocab.json', 'r') as f:
        vocab = json.load(f)
    stoi = vocab['stoi']
    itos = {int(k): v for k, v in vocab['itos'].items()}
    vocab_size = len(stoi)
except FileNotFoundError:
    print("Error: vocab.json not found. Did you run train.py first?")
    sys.exit(1)

encode = lambda s: [stoi.get(c, stoi[' ']) for c in s]
decode = lambda l: ''.join([itos.get(i, '?') for i in l])

# Load the fine-tuned model
print("Loading fine-tuned model...")
model = LanguageModel(vocab_size=vocab_size)
try:
    model.load_state_dict(torch.load('finetuned_model.pt', map_location=device))
    print("Perfectly fine-tuned weights loaded!")
except FileNotFoundError:
    print("Error: finetuned_model.pt not found. Did you run finetune.py first?")
    sys.exit(1)

model = model.to(device)
model.eval()

print("="*50)
print("Welcome to your custom Foundational Model AI!")
print("Type 'exit' or 'quit' to exit.")
print("Example prompt: 'Q: What is a foundation model?'")
print("="*50)

# Provide a prompt for testing directly
default_prompt = "Q: What is a foundation model?\nA:"

def generate_response(prompt_text, max_new_tokens=100):
    # Wrap human input in the Q: A: format the model was fine-tuned on
    formatted_prompt = f"Q: {prompt_text}\nA:"
    
    context = torch.tensor((encode(formatted_prompt)), dtype=torch.long, device=device).unsqueeze(0)
    
    generated_indices = model.generate(context, max_new_tokens=max_new_tokens)
    
    # decode the entire sequence
    full_sequence = decode(generated_indices[0].tolist())
    
    # Extract just the response generated after the prompt
    generated_part = full_sequence[len(formatted_prompt):]
    
    # Simple heuristic to stop generating after it outputs an answer (at the next newline or Q:)
    stop_idx = generated_part.find('\n')
    if stop_idx == -1:
        stop_idx = generated_part.find('Q:')
        
    if stop_idx != -1:
        generated_part = generated_part[:stop_idx]
        
    return generated_part.strip()

if len(sys.argv) > 1 and sys.argv[1] == '--test':
    # Automated test mode
    test_q = "What is a foundation model?"
    print(f"\nUser: {test_q}")
    output = generate_response(test_q)
    print(f"AI: {output}")
else:
    # Interactive chat loop
    while True:
        try:
            prompt = input("\nYou: ")
            if prompt.lower() in ['exit', 'quit']:
                break
            if not prompt.strip():
                continue
                
            print("AI: ", end="", flush=True)
            output = generate_response(prompt)
            print(output)
            
        except KeyboardInterrupt:
            break
            
print("\nExiting. Goodbye!")