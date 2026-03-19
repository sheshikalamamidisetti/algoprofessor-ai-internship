# ============================================================
# CLAUDE CLIENT
# Day 6 Healthcare Bot: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Anthropic Claude API client for data analytics tasks
# ============================================================

# Claude is Anthropic's large language model. It is well suited
# for data analytics tasks because it follows complex instructions
# accurately and produces structured text with good reasoning.
# This file wraps the Anthropic Messages API in a client class
# that mirrors the interface of OpenAIClient so both can be used
# interchangeably in the analytics pipeline. The client falls
# back to mock responses when no API key is set.
# Set your key with:
#   Windows : set ANTHROPIC_API_KEY=sk-ant-...
#   Linux   : export ANTHROPIC_API_KEY=sk-ant-...

import os
import json
import time
import urllib.request
import urllib.error

from data_loader import load_titanic, summarize_dataframe
from openai_client import OpenAIClient

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
DEFAULT_MODEL      = "claude-3-haiku-20240307"
API_VERSION        = "2023-06-01"


# ============================================================
# CLAUDE CLIENT CLASS
# ============================================================

class ClaudeClient:
    """
    Wrapper around the Anthropic Claude Messages API.
    Falls back to mock responses when ANTHROPIC_API_KEY
    is not set in the environment.

    Supported methods:
        chat()     - multi-turn conversation
        complete() - single-turn completion
        analyze()  - data analysis with structured output

    Available models (as of early 2025):
        claude-3-haiku-20240307   - fastest, lowest cost
        claude-3-sonnet-20240229  - balanced speed and quality
        claude-3-opus-20240229    - most capable, highest cost
    """

    def __init__(self, api_key=None, model=DEFAULT_MODEL):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self.model   = model

        if self.api_key:
            print("Claude client ready. Model: " + self.model)
        else:
            print("ANTHROPIC_API_KEY not set. Using mock mode.")
            print("Set key: set ANTHROPIC_API_KEY=sk-ant-your-key-here")

    def chat(self, messages, system_prompt=None, max_tokens=500):
        """
        Sends messages to the Claude Messages API.
        The Anthropic API separates system prompt from messages
        unlike OpenAI which puts system as a message role.

        Parameters:
            messages      : list of dicts with role and content keys
                            roles must be user or assistant only
            system_prompt : string or None, passed separately to API
            max_tokens    : int, maximum tokens in the response

        Returns:
            string response from Claude
        """
        if not self.api_key:
            last = messages[-1]["content"] if messages else ""
            return self._mock_response(last)

        try:
            clean_messages = [
                m for m in messages
                if m.get("role") in ("user", "assistant")
            ]

            body = {
                "model"     : self.model,
                "max_tokens": max_tokens,
                "messages"  : clean_messages
            }
            if system_prompt:
                body["system"] = system_prompt

            payload = json.dumps(body).encode("utf-8")

            request = urllib.request.Request(
                ANTHROPIC_BASE_URL + "/messages",
                data    = payload,
                headers = {
                    "Content-Type"      : "application/json",
                    "x-api-key"         : self.api_key,
                    "anthropic-version" : API_VERSION
                },
                method = "POST"
            )
            response = urllib.request.urlopen(request, timeout=30)
            data     = json.loads(response.read())
            return data["content"][0]["text"].strip()

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            print("Claude HTTP error " + str(e.code) + ": " + error_body[:200])
            last = messages[-1]["content"] if messages else ""
            return self._mock_response(last)
        except Exception as e:
            print("Claude request failed: " + str(e))
            last = messages[-1]["content"] if messages else ""
            return self._mock_response(last)

    def complete(self, prompt, system_prompt=None, max_tokens=500):
        """
        Sends a single user prompt to Claude with optional system prompt.

        Parameters:
            prompt        : string, the user message
            system_prompt : string or None
            max_tokens    : int

        Returns:
            string response from Claude
        """
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, system_prompt=system_prompt,
                         max_tokens=max_tokens)

    def analyze(self, data_summary, question, output_format=None):
        """
        Specialized method for data analysis questions.
        Wraps the data summary and question into a well-structured
        prompt with an optional output format instruction.

        Parameters:
            data_summary  : string, output of summarize_dataframe()
            question      : string, the analytical question
            output_format : string or None, e.g. numbered list or JSON

        Returns:
            string response from Claude
        """
        format_instruction = ""
        if output_format:
            format_instruction = "\nFormat your response as: " + output_format

        prompt = (
            "You are analyzing the following dataset:\n\n"
            + data_summary + "\n\n"
            "Question: " + question
            + format_instruction
        )
        system = (
            "You are a precise data analyst. Answer questions using "
            "only the statistics provided. Always cite specific numbers."
        )
        return self.complete(prompt, system_prompt=system)

    def _mock_response(self, prompt):
        """
        Returns a relevant mock response based on prompt keywords.
        Used when ANTHROPIC_API_KEY is not configured.
        """
        prompt_lower = prompt.lower()
        tag = "[Claude mock] "

        if any(w in prompt_lower for w in ["summarize", "summary", "overview"]):
            return (
                tag + "The Titanic dataset contains 891 passenger records. "
                "Survival rate was 38.4 percent overall. The data shows strong "
                "survival disparities by gender: females at 74.2 percent versus "
                "males at 18.9 percent. Passenger class also showed a clear "
                "gradient: class 1 at 63.0 percent, class 2 at 47.3 percent, "
                "and class 3 at 24.2 percent survival rate."
            )
        elif any(w in prompt_lower for w in ["compare", "difference", "versus"]):
            return (
                tag + "Comparison: Female passengers survived at 74.2 percent "
                "compared to 18.9 percent for males, a difference of 55.3 "
                "percentage points. First class passengers survived at 63.0 "
                "percent compared to 24.2 percent in third class. Both gender "
                "and class differences are statistically significant."
            )
        elif any(w in prompt_lower for w in ["predict", "likelihood", "chances", "would"]):
            return (
                tag + "Based on historical survival patterns, a first-class "
                "female passenger had approximately 96.5 percent survival "
                "probability. A third-class male passenger had approximately "
                "13.5 percent survival probability. The combination of gender "
                "and class is the strongest predictor available in this dataset."
            )
        elif any(w in prompt_lower for w in ["report", "section", "write", "findings"]):
            return (
                tag + "Analysis of Titanic passenger data reveals critical "
                "insights into survival determinants. Gender and passenger "
                "class emerged as the primary survival predictors with females "
                "and first-class passengers significantly overrepresented among "
                "survivors. These findings reflect the evacuation protocols "
                "of the era which prioritized women and first-class passengers "
                "when assigning lifeboat access."
            )
        elif any(w in prompt_lower for w in ["recommend", "suggest", "improve"]):
            return (
                tag + "Three recommendations: First, implement equal lifeboat "
                "access regardless of passenger class. Second, ensure total "
                "lifeboat capacity matches full passenger and crew count. "
                "Third, conduct mandatory safety drills for all passengers "
                "within the first 12 hours of departure."
            )
        else:
            return (
                tag + "Response to: " + prompt[:80] + "... "
                "Set ANTHROPIC_API_KEY environment variable for real Claude responses."
            )


# ============================================================
# API COMPARISON UTILITY
# ============================================================

def compare_apis(openai_client, claude_client, prompt):
    """
    Sends the same prompt to both OpenAI and Claude and prints
    the responses side by side with timing for comparison.

    Parameters:
        openai_client : OpenAIClient instance
        claude_client : ClaudeClient instance
        prompt        : string, the prompt to test on both APIs

    Returns:
        dict with openai and claude response strings and times
    """
    print("\n-- API Comparison: OpenAI vs Claude --")
    print("Prompt: " + prompt[:80] + "...")
    print("-" * 55)

    start           = time.time()
    openai_response = openai_client.complete(prompt)
    openai_time     = round(time.time() - start, 3)

    start           = time.time()
    claude_response = claude_client.complete(prompt)
    claude_time     = round(time.time() - start, 3)

    print("\nOpenAI GPT (" + str(openai_time) + "s):")
    print(openai_response[:350])

    print("\nAnthropic Claude (" + str(claude_time) + "s):")
    print(claude_response[:350])

    faster = "OpenAI" if openai_time < claude_time else "Claude"
    print("\nFaster response: " + faster)

    return {
        "openai": {"response": openai_response, "time_s": openai_time},
        "claude": {"response": claude_response, "time_s": claude_time}
    }


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("CLAUDE CLIENT DEMO")
    print("=" * 55)

    claude = ClaudeClient()
    openai = OpenAIClient()
    df     = load_titanic()
    summary = summarize_dataframe(df)

    print("\n-- Dataset Summary --")
    print(summary)

    print("\n-- Claude Data Analysis --")
    response = claude.analyze(
        summary,
        "What were the three strongest factors affecting passenger survival?",
        output_format="numbered list"
    )
    print(response)

    print("\n-- Claude Report Section --")
    report_prompt = (
        "Write a 3-sentence Key Findings section for a data analysis "
        "report based on this dataset:\n\n" + summary
    )
    report = claude.complete(
        report_prompt,
        system_prompt="You are a professional report writer. Be concise and factual."
    )
    print(report)

    print("\n-- Claude Multi-Turn Analysis --")
    messages = [
        {"role": "user",      "content": "How many passengers are in this dataset?"},
        {"role": "assistant", "content": "The dataset contains 891 passenger records."},
        {"role": "user",      "content": "What percentage of them survived?"},
        {"role": "assistant", "content": "38.4 percent of passengers survived, which is 342 out of 891."},
        {"role": "user",      "content": "Which gender had a better survival rate?"}
    ]
    multi_response = claude.chat(
        messages,
        system_prompt="You are a concise data analyst. Answer with specific numbers."
    )
    print("Claude: " + multi_response)

    print("\n-- OpenAI vs Claude Comparison --")
    compare_apis(
        openai, claude,
        "Summarize the key survival patterns in the Titanic dataset in 2 sentences."
    )

    print("\n-- Claude client demo complete --")


if __name__ == "__main__":
    run_demo()
