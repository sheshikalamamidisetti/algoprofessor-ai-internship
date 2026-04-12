"""
dpo_preference_tuning.py  ·  Day 38  ·  Apr 10
-------------------------------------------------
DPO (Direct Preference Optimisation) for tuning Llama 3.1
to prefer data analyst behaviour over generic responses.

DPO uses (prompt, chosen, rejected) triplets — no reward model needed.
chosen  = good data analyst response (precise, uses correct stats)
rejected = bad response (vague, wrong terminology, no code)

Usage:
    python dpo_preference_tuning.py
    python dpo_preference_tuning.py --dry-run
"""

import argparse
from pathlib import Path
from datasets import Dataset

# ── DPO preference pairs ───────────────────────────────────────────────────
# Each entry: prompt + chosen (expert analyst) + rejected (bad response)

DPO_PAIRS = [
    {
        "prompt": "My ARIMA model residuals show significant autocorrelation. What should I do?",
        "chosen": (
            "Significant residual autocorrelation means the model hasn't captured all patterns. "
            "Steps: 1) Plot ACF/PACF of residuals. 2) If ACF shows spikes at specific lags, "
            "increase the MA order (q). 3) Run Ljung-Box test — if p<0.05, model is inadequate. "
            "4) Consider SARIMA if seasonal patterns remain, or ARCH/GARCH if variance is time-varying. "
            "5) Validate: good residuals should be white noise with zero mean."
        ),
        "rejected": (
            "You should try a different model. Maybe use a neural network instead. "
            "ARIMA is old and there are better options available now."
        ),
    },
    {
        "prompt": "How do I evaluate my time series forecast model?",
        "chosen": (
            "Use these metrics in order: 1) MAE (Mean Absolute Error) — interpretable in original units. "
            "2) RMSE — penalises large errors more heavily. 3) MAPE — percentage error, useful for comparison. "
            "4) sMAPE — symmetric version, avoids division by zero. "
            "For proper evaluation: split into train/val/test using time-based split (never random shuffle). "
            "Use walk-forward validation for robust estimates. "
            "Compare against naive baselines: last-value, seasonal naive, moving average."
        ),
        "rejected": (
            "You can use accuracy or R-squared to evaluate your model. "
            "Higher is better for both metrics."
        ),
    },
    {
        "prompt": "Should I use StandardScaler or MinMaxScaler for my neural network features?",
        "chosen": (
            "Depends on your data and architecture: "
            "StandardScaler (zero mean, unit std): preferred for most neural networks — "
            "handles outliers better, works well with ReLU activations and Adam optimiser. "
            "MinMaxScaler (0-1 range): use when features need bounded range, "
            "or for output layer in regression when target is bounded. "
            "For time series: StandardScaler on features, but scale target separately "
            "and inverse_transform predictions. Always fit on training set only — "
            "never fit on test data to avoid data leakage."
        ),
        "rejected": (
            "Both scalers work fine. Just pick one and it should be okay. "
            "Scaling doesn't matter much for neural networks."
        ),
    },
    {
        "prompt": "My Random Forest feature importances show 3 features dominating. What next?",
        "chosen": (
            "Top-3 features dominating is common and informative. Next steps: "
            "1) Validate with permutation importance — more reliable than impurity-based. "
            "2) Check for multicollinearity among top features (correlation matrix, VIF). "
            "3) Try removing low-importance features (importance < 0.01) and retrain — "
            "often improves generalization. "
            "4) Use SHAP values for instance-level explanations. "
            "5) For time series: ensure top features aren't leaking future information."
        ),
        "rejected": (
            "That means only 3 features matter. You can delete all the other features. "
            "Your model will be much simpler and faster."
        ),
    },
]


def build_dpo_dataset(pairs: list[dict]) -> Dataset:
    """Format for TRL DPOTrainer."""
    return Dataset.from_list([
        {
            "prompt":   p["prompt"],
            "chosen":   p["chosen"],
            "rejected": p["rejected"],
        }
        for p in pairs
    ])


def train_dpo(model_name: str = "outputs/sft_llama31/adapter",
              dry_run: bool = False):
    from trl import DPOTrainer, DPOConfig
    from lora_qlora_setup import load_model_4bit
    from peft import PeftModel
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print("=" * 60)
    print("Day 38 — DPO Preference Tuning for Data Analyst Behaviour")
    print("=" * 60)

    dataset = build_dpo_dataset(DPO_PAIRS)
    print(f"DPO pairs: {len(dataset)}")
    print(f"Sample prompt: {dataset[0]['prompt']}")

    if dry_run:
        print("\nDRY RUN — dataset validated. Run without --dry-run to train.")
        print(f"chosen preview:   {dataset[0]['chosen'][:150]}...")
        print(f"rejected preview: {dataset[0]['rejected'][:150]}...")
        return

    # Load SFT-trained model as starting point
    print(f"\nLoading model from: {model_name}")
    model, tokeniser = load_model_4bit(model_name)

    dpo_config = DPOConfig(
        output_dir="outputs/dpo_llama31",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        beta=0.1,               # DPO temperature — how strongly to prefer chosen
        max_length=512,
        max_prompt_length=256,
        report_to="wandb",
        run_name="day38-dpo-data-analyst",
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,         # None = use SFT model as implicit reference
        args=dpo_config,
        train_dataset=dataset,
        tokenizer=tokeniser,
    )

    trainer.train()

    model.save_pretrained("outputs/dpo_llama31/adapter")
    tokeniser.save_pretrained("outputs/dpo_llama31/adapter")
    print("DPO adapter saved to outputs/dpo_llama31/adapter/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 38 — DPO Preference Tuning")
    parser.add_argument("--model",   default="outputs/sft_llama31/adapter")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_dpo(args.model, args.dry_run)
