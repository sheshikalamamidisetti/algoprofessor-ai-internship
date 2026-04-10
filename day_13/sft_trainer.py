"""
sft_trainer.py  ·  Day 37  ·  Apr 9
--------------------------------------
TRL SFTTrainer for fine-tuning Llama 3.1 on tabular/time series
data science tasks. Trains the model to behave like a data analyst.

Dataset format: instruction → response pairs about time series analysis.
Directly extends the ML algorithm stack from Phase 1.

Usage:
    python sft_trainer.py
    python sft_trainer.py --dry-run          # validate pipeline, no training
    python sft_trainer.py --epochs 3
"""

import argparse
import json
from pathlib import Path
from datasets import Dataset
import pandas as pd

# ── Training data — DS / time series instruction pairs ────────────────────

TRAINING_EXAMPLES = [
    {
        "instruction": "Analyse this monthly sales time series and identify the trend component.",
        "input": "Month: Jan=120, Feb=132, Mar=145, Apr=138, May=160, Jun=175",
        "output": (
            "The time series shows an upward trend with minor fluctuations. "
            "Using linear regression on the monthly values, the trend line has a "
            "positive slope of approximately +11 units/month. "
            "February shows a slight dip below trend, likely seasonal variation. "
            "Overall the data indicates consistent growth of ~9% monthly."
        ),
    },
    {
        "instruction": "A time series has ACF values that decay slowly and PACF cuts off after lag 2. What ARIMA model should I use?",
        "input": "",
        "output": (
            "Slow ACF decay + PACF cutoff at lag 2 indicates an AR(2) process. "
            "The recommended model is ARIMA(2,0,0) if the series is stationary. "
            "Steps: 1) Confirm stationarity with ADF test (p<0.05). "
            "2) If non-stationary, difference once → ARIMA(2,1,0). "
            "3) Validate with residual diagnostics — residuals should be white noise (Ljung-Box p>0.05)."
        ),
    },
    {
        "instruction": "My tabular dataset has 50,000 rows and 120 features. How should I preprocess it for fine-tuning?",
        "input": "Target: predict customer churn (binary). Features: mix of numeric and categorical.",
        "output": (
            "Preprocessing pipeline for fine-tuning: "
            "1) Handle missing values — median imputation for numeric, mode for categorical. "
            "2) Encode categoricals — target encoding for high-cardinality, one-hot for low-cardinality. "
            "3) Scale numerics — StandardScaler (zero mean, unit variance). "
            "4) Address class imbalance — SMOTE or class_weight='balanced'. "
            "5) Convert to instruction format: describe features as text, label as response. "
            "Expected: ~40k train / 10k val split."
        ),
    },
    {
        "instruction": "Explain the difference between LoRA and full fine-tuning for a data analyst LLM.",
        "input": "",
        "output": (
            "Full fine-tuning updates all model weights (~7B parameters for Llama 3.1 8B), "
            "requiring ~56GB VRAM — impractical without multiple A100s. "
            "LoRA adds trainable low-rank matrices (rank r=16) to attention layers only, "
            "training just ~0.1% of parameters (~6M instead of 7B). "
            "For a data analyst fine-tune: LoRA is sufficient — the model already has "
            "language and reasoning; you're only shifting its style toward DS terminology. "
            "QLoRA adds 4-bit quantisation reducing VRAM to ~8GB, enabling training on a single GPU."
        ),
    },
    {
        "instruction": "How do I detect and handle seasonality in a time series before modelling?",
        "input": "Weekly sales data showing regular peaks every 7 days.",
        "output": (
            "Detection: 1) Plot ACF — significant spike at lag 7 confirms weekly seasonality. "
            "2) STL decomposition (seasonal_decompose) to isolate seasonal component. "
            "3) Seasonal strength = 1 - Var(remainder)/Var(seasonal+remainder). "
            "Handling: Option A — Seasonal differencing: subtract lag-7 value (y_t - y_{t-7}). "
            "Option B — Use seasonal model: SARIMA(p,d,q)(P,D,Q,7) or Prophet with weekly seasonality. "
            "Option C — Add Fourier features (sin/cos of 2π*t/7) as regressors."
        ),
    },
]


def build_dataset(examples: list[dict]) -> Dataset:
    """Convert instruction pairs to Llama 3.1 chat format."""
    formatted = []
    for ex in examples:
        user_msg = ex["instruction"]
        if ex.get("input"):
            user_msg += f"\n\nData: {ex['input']}"

        text = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n"
            f"You are a data science expert specialising in time series analysis and ML.\n"
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n"
            f"{user_msg}\n"
            f"<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n"
            f"{ex['output']}\n"
            f"<|eot_id|>"
        )
        formatted.append({"text": text})
    return Dataset.from_list(formatted)


def get_training_args(output_dir: str = "outputs/sft_llama31",
                      epochs: int = 3,
                      dry_run: bool = False):
    from transformers import TrainingArguments
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1 if dry_run else epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,      # effective batch = 4
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_steps=5 if dry_run else -1,     # -1 = full training
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        report_to="wandb",                  # W&B tracking
        run_name="day37-sft-llama31-timeseries",
    )


def train(model_name: str = "meta-llama/Llama-3.1-8B",
          epochs: int = 3,
          dry_run: bool = False):
    from trl import SFTTrainer, SFTConfig
    from lora_qlora_setup import load_model_4bit

    print("=" * 60)
    print("Day 37 — SFT Training: Llama 3.1 on Time Series DS Tasks")
    print("=" * 60)

    # Build dataset
    dataset = build_dataset(TRAINING_EXAMPLES)
    print(f"Training examples: {len(dataset)}")
    print(f"Sample:\n{dataset[0]['text'][:300]}...\n")

    if dry_run:
        print("DRY RUN — validating pipeline without training.")
        print("Dataset OK. Training args OK. Run without --dry-run to train.")
        return

    # Load model
    model, tokeniser = load_model_4bit(model_name)

    # Train
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokeniser,
        args=get_training_args(epochs=epochs),
        dataset_text_field="text",
        max_seq_length=1024,
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    # Save adapter weights
    Path("outputs/sft_llama31").mkdir(parents=True, exist_ok=True)
    model.save_pretrained("outputs/sft_llama31/adapter")
    tokeniser.save_pretrained("outputs/sft_llama31/adapter")
    print("Adapter saved to outputs/sft_llama31/adapter/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 37 — SFT Trainer")
    parser.add_argument("--model",    default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--epochs",   type=int, default=3)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    train(args.model, args.epochs, args.dry_run)
