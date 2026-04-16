import modal

app = modal.App("lora-sql-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "peft", "trl",
        "datasets", "accelerate", "bitsandbytes",
        "huggingface_hub"
    )
)

vol = modal.Volume.from_name("lora-outputs", create_if_missing=True)

@app.function(
    gpu="A10G",          # ← FASTER GPU (3x faster than T4)
    image=image,
    timeout=3600,        # ← 1 hour is enough now
    volumes={"/outputs": vol},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train():
    import torch
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, BitsAndBytesConfig,
        DataCollatorForLanguageModeling,
    )
    from peft import (
        LoraConfig, get_peft_model,
        TaskType, prepare_model_for_kbit_training
    )
    from trl import SFTTrainer
    from datasets import load_dataset

    MODEL_NAME  = "Qwen/Qwen2.5-1.5B"
    OUTPUT_DIR  = "/outputs/lora-sql"
    LORA_RANK   = 8
    LORA_ALPHA  = 16
    TARGET_MODS = ["q_proj", "k_proj", "v_proj", "o_proj"]
    MAX_SEQ_LEN = 256        # ← shorter sequences = faster
    BATCH_SIZE  = 8          # ← bigger batch = faster
    GRAD_ACCUM  = 2
    LR          = 2e-4
    EPOCHS      = 1
    MAX_SAMPLES = 5000       # ← only 5000 samples instead of 74000

    # ── Step 1: Tokenizer ────────────────────────────────────────────────────
    print("Step 1: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Step 2: Model ────────────────────────────────────────────────────────
    print("Step 2: Loading model with 4-bit quantization...")
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # ── Step 3: LoRA ─────────────────────────────────────────────────────────
    print("Step 3: Attaching LoRA adapters...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODS,
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable params: {trainable:,} / Total: {total:,} ({100*trainable/total:.2f}%)")

    # ── Step 4: Dataset (small subset) ───────────────────────────────────────
    print(f"Step 4: Loading dataset (first {MAX_SAMPLES} samples)...")
    raw = load_dataset("b-mc2/sql-create-context", split="train")
    raw = raw.select(range(MAX_SAMPLES))   # ← only 5000 rows

    def fmt(ex):
        return {
            "text": (
                f"### Context:\n{ex['context']}\n\n"
                f"### Question:\n{ex['question']}\n\n"
                f"### SQL:\n{ex['answer']}"
            )
        }

    dataset = raw.map(fmt).train_test_split(test_size=0.1, seed=42)
    print(f"Train: {len(dataset['train'])} | Eval: {len(dataset['test'])}")

    # ── Step 5: Tokenize ─────────────────────────────────────────────────────
    print("Step 5: Tokenizing dataset...")
    def tokenize(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
        )

    tok_train = dataset["train"].map(
        tokenize, batched=True,
        remove_columns=dataset["train"].column_names
    )
    tok_eval = dataset["test"].map(
        tokenize, batched=True,
        remove_columns=dataset["test"].column_names
    )
    tok_train.set_format("torch")
    tok_eval.set_format("torch")

    # ── Step 6: Train ────────────────────────────────────────────────────────
    print("Step 6: Starting training...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=False,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        warmup_steps=20,
        lr_scheduler_type="cosine",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_train,
        eval_dataset=tok_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    vol.commit()
    print("✅ TRAINING COMPLETE! Model saved to", OUTPUT_DIR)

@app.local_entrypoint()
def main():
    train.remote()