"""
model_registry.py  ·  Day 31
-----------------------------
Unified wrapper for GPT-4o, Claude, Gemini, Llama.
All models share one .chat() interface — benchmark code is model-agnostic.

Usage:
    python model_registry.py
"""

import os, time
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

ModelName = Literal["gpt4o", "claude", "gemini", "llama"]


@dataclass
class ModelResponse:
    model: str
    prompt: str
    response: str
    tokens: int | None = None
    latency_ms: float | None = None


class GPT4oModel:
    name = "gpt4o"
    def chat(self, prompt: str, system: str = "You are a data science expert.") -> ModelResponse:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        t0 = time.time()
        r = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return ModelResponse("gpt4o", prompt, r.choices[0].message.content,
                             r.usage.total_tokens, round((time.time()-t0)*1000, 1))


class ClaudeModel:
    name = "claude"
    def chat(self, prompt: str, system: str = "You are a data science expert.") -> ModelResponse:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        t0 = time.time()
        r = client.messages.create(
            model="claude-3-5-sonnet-20241022", max_tokens=1024, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return ModelResponse("claude", prompt, r.content[0].text,
                             r.usage.input_tokens + r.usage.output_tokens,
                             round((time.time()-t0)*1000, 1))


class GeminiModel:
    name = "gemini"
    def chat(self, prompt: str, system: str = "You are a data science expert.") -> ModelResponse:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        t0 = time.time()
        r = genai.GenerativeModel("gemini-1.5-flash").generate_content(f"{system}\n\n{prompt}")
        return ModelResponse("gemini", prompt, r.text, None, round((time.time()-t0)*1000, 1))


class LlamaModel:
    name = "llama"
    def chat(self, prompt: str, system: str = "You are a data science expert.") -> ModelResponse:
        import ollama
        t0 = time.time()
        r = ollama.chat(model="llama3",
                        messages=[{"role": "system", "content": system},
                                  {"role": "user", "content": prompt}])
        return ModelResponse("llama", prompt, r["message"]["content"],
                             None, round((time.time()-t0)*1000, 1))


_REGISTRY = {"gpt4o": GPT4oModel, "claude": ClaudeModel,
             "gemini": GeminiModel, "llama": LlamaModel}


def get_model(name: ModelName):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose: {list(_REGISTRY)}")
    return _REGISTRY[name]()


def load_all_models() -> dict:
    out = {}
    for name, cls in _REGISTRY.items():
        try:
            out[name] = cls()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")
    return out


if __name__ == "__main__":
    print("Loading all models...")
    models = load_all_models()
    test = "What is a p-value? One sentence."
    for name, model in models.items():
        r = model.chat(test)
        print(f"\n{name}: {r.response[:120]}  [{r.latency_ms}ms]")
