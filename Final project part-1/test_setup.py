from datasets import load_dataset
from transformers import AutoTokenizer

print("Testing dataset download...")
ds = load_dataset("b-mc2/sql-create-context", split="train[:10]")
print(f"Dataset OK - {len(ds)} samples")
print("Sample question:", ds[0]['question'])

print("\nTesting tokenizer download...")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B", trust_remote_code=True)
print("Tokenizer OK")
print("\nAll checks passed! Ready to train on Modal.")