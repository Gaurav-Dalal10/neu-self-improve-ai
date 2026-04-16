# LoRA Fine-tuning: Text-to-SQL

Fine-tune `Qwen/Qwen2.5-1.5B` for Text-to-SQL generation using LoRA, then run inference and evaluation.

## Model

- Base model: `Qwen/Qwen2.5-1.5B`
- Fine-tuning method: LoRA (`r=8`, `alpha=16`, `dropout=0.05`)
- Task: Text-to-SQL
- Trainable parameters: `2,179,072 / 1,545,893,376 (0.14%)`

## Results

| Metric           | Value         |
| ---------------- | ------------- |
| Final Train Loss | 0.93          |
| Final Eval Loss  | 0.85          |
| Training Time    | 533 seconds   |
| GPU              | A10G on Modal |

## Project Files

- `train_lora.py`: Local LoRA training (4-bit quantization)
- `modal_train.py`: Cloud training on Modal (A10G)
- `inference.py`: Load base model + LoRA checkpoint and generate SQL
- `eval/evaluate.py`: Exact-match evaluation
- `data/prepare_data.py`: Dataset formatting helper
- `test_setup.py`: Quick dependency and download checks
- `requirements.txt`: Project dependencies
- `model_download/lora-sql/`: Saved/downloaded LoRA checkpoints
- `outputs/lora-sql/`: Local training outputs

## Prerequisites

1. Python `3.10`, `3.11`, or `3.12`
2. Hugging Face account and access to required model/dataset
3. Optional for local training: NVIDIA GPU (recommended for `bitsandbytes`)

## Setup

Use one environment option only.

### Option A: Conda

```bash
conda create -n lora_env python=3.11 -y
conda activate lora_env
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option B: Windows venv (PowerShell)

```powershell
python -m venv lora_env
.\lora_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Important:

- In PowerShell, use `Activate.ps1`.
- Do not run `...\Scripts\activate` directly in PowerShell.

## Quick Validation

```bash
python test_setup.py
```

Expected messages:

- `Dataset OK - ... samples`
- `Tokenizer OK`
- `All checks passed!`

## Training

### Local Training

```bash
python train_lora.py
```

Current defaults in `train_lora.py`:

- Base model: `Qwen/Qwen2.5-1.5B`
- Dataset: `b-mc2/sql-create-context`
- Output dir: `./outputs/lora-sql`
- Epochs: `3`
- Learning rate: `2e-4`
- Batch size: `4`, grad accumulation: `4`

### Modal Cloud Training

1. Install and configure Modal:

```bash
python -m pip install modal
modal setup
```

2. Create the required secret:

```bash
modal secret create huggingface-secret HF_TOKEN=<your_hf_token>
```

3. Run training:

```bash
modal run modal_train.py
```

## Inference

```bash
python inference.py
```

Current defaults in `inference.py`:

- Base model: `Qwen/Qwen2.5-1.5B`
- LoRA checkpoint: `./model_download/lora-sql/checkpoint-282`
- Device: CPU (`device_map="cpu"`)

If your best checkpoint is different, update the `LORA` path in `inference.py`.

## Sample Inference Output (Your Run)

```text
--- Test 1 ---
Question : What is the average salary by department?
SQL      : SELECT dept, AVG(salary) FROM employees GROUP BY dept ORDER BY AVG(salary)

--- Test 2 ---
Question : How many students passed with grade above 50?
SQL      : SELECT COUNT(*) FROM students WHERE grade > 50 AND subject = "Math"

--- Test 3 ---
Question : What is the total amount spent by each customer?
SQL      : SELECT customer, SUM(amount) FROM orders GROUP BY customer ORDER BY SUM(amount)
```

## Evaluation

```bash
python eval/evaluate.py
```

This prints exact-match accuracy:

- `Exact Match Accuracy: <score> (<correct>/<n>)`

To change sample count, update `run_eval(n_samples=200)` in `eval/evaluate.py`.

## Troubleshooting

1. PowerShell activation error
- Use `.\lora_env\Scripts\Activate.ps1`.

2. `ModuleNotFoundError` (for example `peft`)
- Activate your environment and reinstall dependencies:

```bash
python -m pip install -r requirements.txt
```

3. `bitsandbytes` issues on CPU-only/Windows
- Prefer inference locally, or use `modal_train.py` for GPU training.

## Suggested Run Order

1. `python test_setup.py`
2. `python inference.py`
3. `python train_lora.py` or `modal run modal_train.py`
4. Update `LORA` path in `inference.py` if needed
5. `python eval/evaluate.py`

## Dependencies

From `requirements.txt`:

- `torch`
- `transformers`
- `peft`
- `datasets`
- `accelerate`
- `bitsandbytes`
- `trl`
- `wandb`
- `modal`
- `huggingface_hub`
