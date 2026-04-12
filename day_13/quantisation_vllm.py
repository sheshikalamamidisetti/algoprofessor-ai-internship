"""
quantisation_vllm.py  ·  Day 39  ·  Apr 11
--------------------------------------------
GPTQ/AWQ quantisation of fine-tuned model + vLLM for fast inference.
Converts the DPO-tuned adapter to a deployable quantised model.

GPTQ: post-training quantisation, 4-bit, good accuracy
AWQ : activation-aware quantisation, 4-bit, faster inference

Usage:
    python quantisation_vllm.py --mode gptq
    python quantisation_vllm.py --mode awq
    python quantisation_vllm.py --mode vllm   # run inference server
    python quantisation_vllm.py --mode demo   # demo without GPU
"""

import argparse
from pathlib import Path


# ── GPTQ Quantisation ──────────────────────────────────────────────────────

def quantise_gptq(model_path: str = "outputs/dpo_llama31/adapter",
                  output_path: str = "outputs/llama31_gptq"):
    """Quantise fine-tuned model to GPTQ 4-bit."""
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from datasets import load_dataset

    print("GPTQ 4-bit quantisation...")

    tokeniser = AutoTokenizer.from_pretrained(model_path)

    # Calibration data — use our DS training examples
    calibration_texts = [
        "Analyse this time series and identify seasonality: Jan=100 Feb=95 Mar=110",
        "What ARIMA order should I use if ACF decays slowly and PACF cuts at lag 2?",
        "Explain the difference between MAE and RMSE for time series evaluation.",
        "How do I handle missing values in a pandas DataFrame before training?",
        "What is the difference between LoRA and full fine-tuning?",
    ]
    examples = [tokeniser(t, return_tensors="pt") for t in calibration_texts]

    quantise_config = BaseQuantizeConfig(
        bits=4,
        group_size=128,
        desc_act=False,
    )

    model = AutoGPTQForCausalLM.from_pretrained(model_path, quantize_config=quantise_config)
    model.quantize(examples)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_quantized(output_path)
    tokeniser.save_pretrained(output_path)
    print(f"GPTQ model saved → {output_path}")


# ── AWQ Quantisation ───────────────────────────────────────────────────────

def quantise_awq(model_path: str = "outputs/dpo_llama31/adapter",
                 output_path: str = "outputs/llama31_awq"):
    """Quantise fine-tuned model to AWQ 4-bit."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    print("AWQ 4-bit quantisation...")

    tokeniser = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    # Calibration
    calib_data = [
        "Time series forecasting with ARIMA requires stationarity.",
        "LoRA fine-tuning adds low-rank matrices to attention layers.",
        "Cross-validation for time series uses walk-forward splits.",
    ]
    model.quantize(tokeniser, quant_config=quant_config, calib_data=calib_data)

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_quantized(output_path)
    tokeniser.save_pretrained(output_path)
    print(f"AWQ model saved → {output_path}")


# ── vLLM Inference ─────────────────────────────────────────────────────────

def run_vllm_inference(model_path: str = "outputs/llama31_awq"):
    """Run fast inference with vLLM on quantised model."""
    from vllm import LLM, SamplingParams

    print(f"Loading {model_path} with vLLM...")
    llm = LLM(
        model=model_path,
        quantization="awq",
        dtype="float16",
        max_model_len=2048,
        gpu_memory_utilization=0.85,
    )

    sampling = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=512,
    )

    # DS inference prompts
    prompts = [
        "Analyse this monthly revenue data and forecast next month: Jan=50k Feb=55k Mar=60k Apr=58k May=65k",
        "My LSTM model on time series has high training loss. What are the top 3 causes?",
        "Compare ARIMA vs Prophet for weekly sales forecasting.",
    ]

    print("\nRunning inference...")
    outputs = llm.generate(prompts, sampling)
    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt[:80]}...")
        print(f"Response: {output.outputs[0].text[:400]}")
        print("-" * 60)


# ── Demo (no GPU needed) ───────────────────────────────────────────────────

def run_demo():
    """Show what quantisation does — no model loading needed."""
    print("=" * 60)
    print("Day 39 — Quantisation & vLLM Demo")
    print("=" * 60)
    print()
    print("GPTQ vs AWQ comparison:")
    print()

    data = [
        ("Method",     "GPTQ",          "AWQ"),
        ("Bits",       "4-bit",         "4-bit"),
        ("Approach",   "weight quant",  "activation-aware"),
        ("Speed",      "moderate",      "faster inference"),
        ("Accuracy",   "good",          "slightly better"),
        ("Use case",   "general",       "throughput-critical"),
        ("VRAM saved", "~4x vs fp16",   "~4x vs fp16"),
    ]

    col_w = [18, 20, 24]
    header = f"{'Property':<{col_w[0]}} {'GPTQ':<{col_w[1]}} {'AWQ':<{col_w[2]}}"
    print(header)
    print("-" * sum(col_w))
    for row in data[1:]:
        print(f"{row[0]:<{col_w[0]}} {row[1]:<{col_w[1]}} {row[2]:<{col_w[2]}}")

    print()
    print("vLLM throughput vs HuggingFace generate():")
    print("  HuggingFace: ~20-50 tokens/sec (single request)")
    print("  vLLM:        ~200-500 tokens/sec (continuous batching)")
    print("  Speedup:     ~10x for concurrent requests")
    print()
    print("Pipeline for TimeSeriesHunter deployment:")
    print("  1. Train:    SFTTrainer + DPO  →  adapter weights")
    print("  2. Quantise: AWQ 4-bit          →  llama31_awq/")
    print("  3. Serve:    vLLM               →  localhost:8000/v1")
    print("  4. Track:    W&B                →  wandb.ai dashboard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 39 — Quantisation + vLLM")
    parser.add_argument("--mode", choices=["gptq", "awq", "vllm", "demo"],
                        default="demo")
    parser.add_argument("--model", default="outputs/dpo_llama31/adapter")
    args = parser.parse_args()

    modes = {
        "gptq": lambda: quantise_gptq(args.model),
        "awq":  lambda: quantise_awq(args.model),
        "vllm": run_vllm_inference,
        "demo": run_demo,
    }
    modes[args.mode]()
