import json
import random

def generate_qa_dataset(num_samples=5000):
    topics = {
        "AI": ["artificial intelligence", "machine learning", "neural networks", "models"],
        "Python": ["programming language", "coding", "scripting", "development"],
        "PyTorch": ["deep learning framework", "tensors", "neural network building", "AI training"],
        "Transformer": ["attention mechanism", "language model architecture", "self-attention", "GPT"],
        "Dataset": ["collection of data", "training information", "raw facts", "input for models"],
        "Fine-tuning": ["process of alignment", "specializing a model", "improving accuracy", "model training"],
        "Pre-training": ["initial training phase", "learning from raw text", "building a base model", "learning grammar"]
    }

    templates = [
        "Q: What is {topic}?\nA: {topic} is a {definition}.\n",
        "Q: Tell me about {topic}.\nA: {topic} represents {definition}.\n",
        "Q: Why use {topic}?\nA: It is useful for {definition}.\n",
        "Q: Define {topic}.\nA: {definition}.\n"
    ]

    dataset = []
    
    # Generate template based QA
    for _ in range(num_samples // 2):
        topic_key = random.choice(list(topics.keys()))
        topic_name = topic_key
        definition = random.choice(topics[topic_key])
        template = random.choice(templates)
        dataset.append(template.format(topic=topic_name, definition=definition))
        
    # Generate Math QA (very basic)
    for _ in range(num_samples - len(dataset)):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        op = random.choice(['+', '-'])
        if op == '+':
            res = a + b
        else:
            res = a - b
        dataset.append(f"Q: What is {a} {op} {b}?\nA: The result is {res}.\n")

    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    print("Generating 5000 QA pairs...")
    data = generate_qa_dataset(5000)
    with open('qa_dataset.json', 'w') as f:
        json.dump(data, f)
    print(f"Dataset saved to qa_dataset.json with {len(data)} entries.")
