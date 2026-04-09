"""
lora_qlora_setup.py  ·  Day 36  ·  Apr 8
------------------------------------------
LoRA and QLoRA setup for Llama 3.1 fine-tuning.
Covers: BitsAndBytes 4-bit quantisation, PEFT LoRA config,
model loading, tokeniser prep.

Usage:
    python lora_qlora_setup.py
    python lora_qlora_setup.py --model meta-llama/Llama-3.1-8B
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ── Config ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"

QLORA_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,       # nested quantisation for memory
    bnb_4bit_quant_type="nf4",            # NormalFloat4 — better for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,
)

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                   # rank — higher = more capacity, more VRAM
    lora_alpha=32,          # scaling factor (alpha/r = 2 is standard)
    target_modules=[        # which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
)


def load_model_4bit(model_name: str = DEFAULT_MODEL):
    """Load model in 4-bit QLoRA mode."""
    print(f"Loading {model_name} in 4-bit QLoRA mode...")

    tokeniser = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokeniser.pad_token = tokeniser.eos_token
    tokeniser.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=QLORA_CONFIG,
        device_map="auto",
        trust_remote_code=True,
    )

    # Prepare for k-bit training — freezes base weights, enables grad checkpointing
    model = prepare_model_for_kbit_training(model)

    # Apply LoRA adapters
    model = get_peft_model(model, LORA_CONFIG)

    print(f"\nModel loaded. Trainable parameters:")
    model.print_trainable_parameters()

    return model, tokeniser


def load_model_lora_only(model_name: str = DEFAULT_MODEL):
    """Load in full precision with LoRA only (no quantisation) — for CPU/MPS."""
    print(f"Loading {model_name} with LoRA (full precision)...")

    tokeniser = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokeniser.pad_token = tokeniser.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return model, tokeniser


def print_memory_stats():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved  = torch.cuda.memory_reserved()  / 1e9
        print(f"GPU memory — allocated: {allocated:.2f} GB | reserved: {reserved:.2f} GB")
    else:
        print("No GPU detected — will use CPU/MPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 36 — LoRA/QLoRA setup")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=["qlora", "lora"], default="qlora")
    args = parser.parse_args()

    print("=" * 60)
    print("Day 36 — LoRA/QLoRA Setup for Llama 3.1")
    print("=" * 60)
    print(f"Mode:  {args.mode}")
    print(f"Model: {args.model}")
    print_memory_stats()

    # Print config summary without loading model
    print("\nLoRA Config:")
    print(f"  rank (r):       {LORA_CONFIG.r}")
    print(f"  alpha:          {LORA_CONFIG.lora_alpha}")
    print(f"  target modules: {LORA_CONFIG.target_modules}")
    print(f"  dropout:        {LORA_CONFIG.lora_dropout}")

    print("\nQLoRA Config:")
    print(f"  bits:           4")
    print(f"  quant type:     nf4")
    print(f"  compute dtype:  bfloat16")
    print(f"  double quant:   True")

    print("\nTo load the model, run:")
    print(f"  from lora_qlora_setup import load_model_4bit")
    print(f"  model, tokeniser = load_model_4bit('{args.model}')")
    print("\nNote: Requires ~8GB VRAM for 8B model in 4-bit mode.")
    print("      Use Google Colab A100/L4 or local GPU.")
