from datasets import load_dataset
from inference import load_model, generate_sql
import re

def exact_match(pred, gold):
    # Normalize whitespace and case for comparison
    norm = lambda s: re.sub(r'\s+', ' ', s.strip().lower())
    return norm(pred) == norm(gold)

def run_eval(n_samples=200):
    model, tok = load_model()
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    dataset = dataset.shuffle(seed=42).select(range(n_samples))
    
    correct = 0
    for ex in dataset:
        pred = generate_sql(model, tok, ex["context"], ex["question"])
        if exact_match(pred, ex["answer"]):
            correct += 1
    
    acc = correct / n_samples
    print(f"Exact Match Accuracy: {acc:.3f} ({correct}/{n_samples})")
    return acc

if __name__ == "__main__":
    run_eval()