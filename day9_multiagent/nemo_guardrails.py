"""
nemo_guardrails.py  ·  Day 46  ·  Apr 22
------------------------------------------
Goes into: day9_multiagent/  (extended)

NeMo Guardrails for safe LLM outputs + Presidio PII redaction
for sensitive data pipelines. Protects InsightBot from:
  - Prompt injection attacks
  - Sensitive data leakage (names, emails, SSNs, credit cards)
  - Off-topic responses in a DS analyst context
  - Hallucinated statistics

Usage:
    python nemo_guardrails.py
    python nemo_guardrails.py --demo
    python nemo_guardrails.py --test-pii
"""

import re
import json
import argparse
from pathlib import Path
from typing import Any

# ── Presidio PII Redaction ─────────────────────────────────────────────────

PII_PATTERNS = {
    "email":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone":       r"\b(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn":         r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address":  r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "name":        r"\b(Mr\.|Mrs\.|Dr\.|Prof\.)\s+[A-Z][a-z]+\s+[A-Z][a-z]+\b",
}

REDACTION_MAP = {
    "email":       "[EMAIL_REDACTED]",
    "phone":       "[PHONE_REDACTED]",
    "ssn":         "[SSN_REDACTED]",
    "credit_card": "[CARD_REDACTED]",
    "ip_address":  "[IP_REDACTED]",
    "name":        "[NAME_REDACTED]",
}


def redact_pii(text: str) -> tuple[str, list[dict]]:
    """Remove PII from text. Returns redacted text + list of what was found."""
    redacted = text
    findings = []
    for entity_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, redacted, re.IGNORECASE)
        if matches:
            findings.append({"type": entity_type, "count": len(matches)})
            redacted = re.sub(
                pattern, REDACTION_MAP[entity_type],
                redacted, flags=re.IGNORECASE
            )
    return redacted, findings


def presidio_redact(text: str) -> tuple[str, list]:
    """Use Microsoft Presidio for production-grade PII detection."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine

        analyzer   = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        results   = analyzer.analyze(text=text, language="en")
        redacted  = anonymizer.anonymize(text=text, analyzer_results=results)
        findings  = [{"type": r.entity_type, "score": r.score} for r in results]
        return redacted.text, findings

    except ImportError:
        print("Presidio not installed — using regex fallback")
        print("Install: pip install presidio-analyzer presidio-anonymizer")
        return redact_pii(text)


# ── NeMo Guardrails Config ─────────────────────────────────────────────────

GUARDRAILS_CONFIG = """
define user ask off-topic question
  "What is the weather today?"
  "Can you write me a poem?"
  "Tell me a joke"

define bot refuse off-topic
  "I am InsightBot, a data science analyst assistant. I can only help with
   data analysis, statistics, machine learning, and business intelligence queries."

define flow off-topic guard
  user ask off-topic question
  bot refuse off-topic

define user share sensitive data
  "Here is my SSN: 123-45-6789"
  "My credit card is 4111 1111 1111 1111"
  "Customer emails: john@company.com, jane@company.com"

define bot handle sensitive data
  "I have detected sensitive data in your message. For security, I cannot
   process personal identifiers. Please anonymise the data before analysis."

define flow pii guard
  user share sensitive data
  bot handle sensitive data

define user ask for made up stats
  "Make up some statistics for my report"
  "Just invent some numbers that look good"

define bot refuse fabrication
  "I cannot fabricate or invent statistics. All figures in InsightBot reports
   are derived from actual data analysis. Please provide real data."

define flow no fabrication
  user ask for made up stats
  bot refuse fabrication
"""


def save_guardrails_config(path: str = "guardrails_config"):
    """Save NeMo Guardrails config files."""
    Path(path).mkdir(exist_ok=True)

    with open(f"{path}/config.co", "w") as f:
        f.write(GUARDRAILS_CONFIG)

    config_yaml = """
models:
  - type: main
    engine: openai
    model: gpt-4o

instructions:
  - type: general
    content: |
      You are InsightBot, a data science analyst assistant.
      Only answer questions about data analysis, statistics,
      machine learning, and business intelligence.
      Never fabricate statistics or analysis results.
      Always flag if sensitive data is detected in inputs.

rails:
  input:
    flows:
      - pii guard
      - off-topic guard
  output:
    flows:
      - no fabrication
"""
    with open(f"{path}/config.yml", "w") as f:
        f.write(config_yaml)

    print(f"Guardrails config saved to {path}/")
    return path


def apply_guardrails(user_message: str, use_nemo: bool = False) -> dict:
    """
    Apply guardrails to a user message.
    Returns safe response or rejection with reason.
    """
    # Step 1: PII check
    redacted_msg, pii_found = redact_pii(user_message)

    if pii_found:
        return {
            "status":   "blocked",
            "reason":   "PII detected",
            "pii_types": [f["type"] for f in pii_found],
            "message":  "Sensitive data detected and redacted. Please anonymise before analysis.",
            "redacted": redacted_msg,
        }

    # Step 2: Off-topic check
    off_topic_keywords = ["weather", "poem", "joke", "recipe", "sports",
                          "movie", "music", "celebrity"]
    if any(kw in user_message.lower() for kw in off_topic_keywords):
        return {
            "status":  "blocked",
            "reason":  "off-topic",
            "message": "InsightBot only handles data science and analytics queries.",
        }

    # Step 3: Fabrication request check
    fabrication_keywords = ["make up", "invent", "fabricate", "fake data",
                            "just create", "generate fake"]
    if any(kw in user_message.lower() for kw in fabrication_keywords):
        return {
            "status":  "blocked",
            "reason":  "fabrication request",
            "message": "Cannot fabricate statistics. Please provide real data.",
        }

    # Step 4: NeMo Guardrails (if enabled)
    if use_nemo:
        try:
            from nemoguardrails import LLMRails, RailsConfig
            config = RailsConfig.from_path("guardrails_config")
            rails  = LLMRails(config)
            response = rails.generate(messages=[{
                "role": "user", "content": redacted_msg
            }])
            return {
                "status":   "allowed",
                "response": response,
                "redacted": redacted_msg != user_message,
            }
        except ImportError:
            print("nemoguardrails not installed — using rule-based fallback")

    return {
        "status":   "allowed",
        "message":  redacted_msg,
        "redacted": redacted_msg != user_message,
    }


def run_demo():
    print("=" * 60)
    print("Day 46 — NeMo Guardrails + Presidio PII Demo")
    print("=" * 60)

    test_cases = [
        ("Normal DS query",
         "Analyse churn rate for customers with tenure < 12 months"),
        ("PII in data",
         "Customer john.doe@company.com has SSN 123-45-6789 and high churn risk"),
        ("Off-topic",
         "What is the weather like in Hyderabad today?"),
        ("Fabrication request",
         "Just make up some good looking statistics for my quarterly report"),
        ("Sensitive DS query",
         "Our revenue data: Q1=50k, Q2=55k, Q3=60k. Forecast Q4."),
    ]

    results = []
    for name, msg in test_cases:
        result = apply_guardrails(msg)
        results.append({"test": name, **result})
        status_symbol = "BLOCKED" if result["status"] == "blocked" else "ALLOWED"
        print(f"\n[{status_symbol}] {name}")
        print(f"  Input:  {msg[:70]}...")
        if result["status"] == "blocked":
            print(f"  Reason: {result['reason']}")
            print(f"  Reply:  {result['message']}")
        else:
            print(f"  PII redacted: {result.get('redacted', False)}")

    # Save results
    Path("outputs").mkdir(exist_ok=True)
    from datetime import datetime
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"outputs/guardrails_demo_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day 46 — Guardrails + PII")
    parser.add_argument("--demo",      action="store_true", default=True)
    parser.add_argument("--test-pii",  action="store_true")
    parser.add_argument("--save-config", action="store_true")
    args = parser.parse_args()

    if args.save_config:
        save_guardrails_config()
    elif args.test_pii:
        test = "Contact Dr. John Smith at john.smith@company.com or 555-123-4567. SSN: 123-45-6789"
        redacted, found = redact_pii(test)
        print(f"Original: {test}")
        print(f"Redacted: {redacted}")
        print(f"Found:    {found}")
    else:
        run_demo()
