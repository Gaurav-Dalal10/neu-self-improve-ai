from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_sql_dataset(model_name="Qwen/Qwen2.5-1.5B"):
    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def format_example(example):
        # Format: given schema + question -> SQL
        prompt = (
            f"### Context:\n{example['context']}\n\n"
            f"### Question:\n{example['question']}\n\n"
            f"### SQL:\n{example['answer']}"
        )
        return {"text": prompt}
    
    dataset = dataset.map(format_example)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return dataset, tokenizer

if __name__ == "__main__":
    ds, tok = prepare_sql_dataset()
    print(ds)
    print(ds["train"][0]["text"])