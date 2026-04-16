import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer
from datasets import load_dataset
import wandb
import yaml
import os

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-1.5B"   # tiny, strong base (used in Tina paper)
DATASET_NAME = "b-mc2/sql-create-context"
OUTPUT_DIR   = "./outputs/lora-sql"
LORA_RANK    = 8          # r — key LoRA hyperparameter
LORA_ALPHA   = 16         # scaling factor
LORA_DROPOUT = 0.05
TARGET_MODS  = ["q_proj", "k_proj", "v_proj", "o_proj"]  # attention layers
MAX_SEQ_LEN  = 512
BATCH_SIZE   = 4
GRAD_ACCUM   = 4          # effective batch = 16
LR           = 2e-4
EPOCHS       = 3

# ── Load tokenizer ───────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ── Quantization (4-bit) for memory efficiency ───────────────────────────────
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ── Load base model ──────────────────────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_cfg,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# ── LoRA Config ──────────────────────────────────────────────────────────────
# This implements: h = W0*x + (B*A)*x
# B initialized to 0, A ~ N(0, sigma^2)
# So delta_W = B*A = 0 at init (no perturbation to pretrained model)
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,           # rank of decomposition
    lora_alpha=LORA_ALPHA, # scaling: delta_W scaled by alpha/r
    target_modules=TARGET_MODS,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    inference_mode=False,
)

model = get_peft_model(model, lora_cfg)

# Print trainable parameter count (should be << total params)
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / Total: {total:,} "
      f"({100*trainable/total:.2f}%)")

# ── Dataset ──────────────────────────────────────────────────────────────────
raw = load_dataset(DATASET_NAME, split="train")

def format_prompt(ex):
    return {
        "text": (
            f"### Context:\n{ex['context']}\n\n"
            f"### Question:\n{ex['question']}\n\n"
            f"### SQL:\n{ex['answer']}"
        )
    }

dataset = raw.map(format_prompt).train_test_split(test_size=0.05, seed=42)

# ── Training Args ─────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    fp16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_steps=200,
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="wandb",
    run_name="lora-sql-qwen",
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,
)

wandb.init(project="lora-assignment", name="lora-sql")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print("Training complete. Model saved to", OUTPUT_DIR)