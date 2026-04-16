import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen2.5-1.5B"
LORA = "./model_download/lora-sql/checkpoint-282"  # ← updated path

def load_model(base=BASE, lora_path=LORA):
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    print("Loading base model (CPU, this takes ~1 min)...")
    model = AutoModelForCausalLM.from_pretrained(
        base,
        torch_dtype=torch.float32,  # ← float32 for CPU
        device_map="cpu",           # ← CPU since no local GPU
        trust_remote_code=True,
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.merge_and_unload()  # merge for zero inference latency
    model.eval()
    print("Model ready!")
    return model, tok

def generate_sql(model, tok, context, question, max_new=100):
    prompt = (
        f"### Context:\n{context}\n\n"
        f"### Question:\n{question}\n\n"
        f"### SQL:\n"
    )
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen.strip()

if __name__ == "__main__":
    model, tok = load_model()

    # ── Test 1 ───────────────────────────────────────────────────────────────
    context  = "CREATE TABLE employees (id INT, name TEXT, dept TEXT, salary INT)"
    question = "What is the average salary by department?"
    sql = generate_sql(model, tok, context, question)
    print("\n--- Test 1 ---")
    print(f"Question : {question}")
    print(f"SQL      : {sql}")

    # ── Test 2 ───────────────────────────────────────────────────────────────
    context  = "CREATE TABLE students (id INT, name TEXT, grade INT, subject TEXT)"
    question = "How many students passed with grade above 50?"
    sql = generate_sql(model, tok, context, question)
    print("\n--- Test 2 ---")
    print(f"Question : {question}")
    print(f"SQL      : {sql}")

    # ── Test 3 ───────────────────────────────────────────────────────────────
    context  = "CREATE TABLE orders (id INT, customer TEXT, amount FLOAT, date TEXT)"
    question = "What is the total amount spent by each customer?"
    sql = generate_sql(model, tok, context, question)
    print("\n--- Test 3 ---")
    print(f"Question : {question}")
    print(f"SQL      : {sql}")