# ============================================================
# DSPY PIPELINE
# Day 5 NLP: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: DSPy-style modular prompt engineering for data analysis
# ============================================================

# DSPy is a framework from Stanford released in 2023 that
# treats prompts as learnable programs rather than hand-written
# strings. The core idea is that you define a Signature which
# specifies input and output field names, then DSPy automatically
# generates and optimizes the prompt to fill those fields.
# When given labeled examples it can also compile the program
# to improve accuracy without manual prompt tuning. This file
# implements DSPy-style modules from scratch so they work
# without the dspy package installed. Each module has a clear
# Signature defining what goes in and what comes out, and the
# pipeline chains multiple modules together in sequence. The
# few-shot demonstration at the end shows how providing
# examples dramatically changes the style and quality of LLM
# output compared to zero-shot prompting.

import pandas as pd
import numpy as np

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False

from ollama_setup import OllamaClient
from llama_pipeline import load_titanic, summarize_dataframe

MODEL = "llama3"


# ============================================================
# SIGNATURE CLASS
# ============================================================

class Signature:
    """
    Defines the contract for a DSPy-style module by specifying
    the task description, required input field names, and
    expected output field names. The Predict and ChainOfThought
    modules use this to build structured prompts automatically.

    Parameters:
        description : plain English description of the task
        inputs      : list of input field name strings
        outputs     : list of output field name strings
    """

    def __init__(self, description, inputs, outputs):
        self.description = description
        self.inputs      = inputs
        self.outputs     = outputs

    def __repr__(self):
        return (
            "Signature(task='" + self.description + "', "
            "inputs=" + str(self.inputs) + ", "
            "outputs=" + str(self.outputs) + ")"
        )


# ============================================================
# PREDICT MODULE
# ============================================================

class Predict:
    """
    DSPy-style Predict module. Takes a Signature and an
    OllamaClient, then when called with keyword arguments
    matching the signature inputs, it builds a structured
    prompt and returns a dict of output field values.

    This is equivalent to dspy.Predict(Signature) in the
    real DSPy library.

    Parameters:
        signature : Signature instance defining inputs and outputs
        client    : OllamaClient instance for LLM calls
        model     : model name string, default is llama3
    """

    def __init__(self, signature, client, model=MODEL):
        self.signature = signature
        self.client    = client
        self.model     = model

    def __call__(self, **kwargs):
        prompt   = self._build_prompt(kwargs)
        response = self.client.generate(self.model, prompt)
        return self._build_output(response)

    def _build_prompt(self, inputs):
        """Constructs prompt from signature description and inputs."""
        lines = [
            "Task: " + self.signature.description,
            ""
        ]
        for field_name in self.signature.inputs:
            value = inputs.get(field_name, "not provided")
            lines.append(field_name.replace("_", " ").title() + ": " + str(value))

        lines.append("")
        lines.append("Provide the following output fields:")
        for field_name in self.signature.outputs:
            lines.append("  " + field_name.replace("_", " ").title() + ":")

        return "\n".join(lines)

    def _build_output(self, response):
        """Returns dict with each output field mapped to the response."""
        return {field: response for field in self.signature.outputs}


# ============================================================
# CHAIN OF THOUGHT MODULE
# ============================================================

class ChainOfThought:
    """
    DSPy-style ChainOfThought module. Extends Predict by adding
    a reasoning step before the final output fields. This is
    equivalent to dspy.ChainOfThought(Signature) in the real
    DSPy library. The model is asked to think step by step
    before filling in the output fields.

    Parameters:
        signature : Signature instance defining inputs and outputs
        client    : OllamaClient instance for LLM calls
        model     : model name string, default is llama3
    """

    def __init__(self, signature, client, model=MODEL):
        self.signature = signature
        self.client    = client
        self.model     = model

    def __call__(self, **kwargs):
        prompt   = self._build_cot_prompt(kwargs)
        response = self.client.generate(self.model, prompt)
        return {"reasoning": response, **{f: response for f in self.signature.outputs}}

    def _build_cot_prompt(self, inputs):
        """Builds a CoT-enhanced prompt from signature and inputs."""
        lines = [
            "Task: " + self.signature.description,
            "Think step by step before providing your answer.",
            ""
        ]
        for field_name in self.signature.inputs:
            value = inputs.get(field_name, "not provided")
            lines.append(field_name.replace("_", " ").title() + ": " + str(value))

        lines.append("")
        lines.append("Step-by-step reasoning:")
        lines.append("")
        lines.append("Final output fields:")
        for field_name in self.signature.outputs:
            lines.append("  " + field_name.replace("_", " ").title() + ":")

        return "\n".join(lines)


# ============================================================
# SPECIALIZED MODULES
# ============================================================

class DataSummarizer:
    """
    DSPy module that reads a raw dataset summary and audience
    type, then generates an executive summary and list of
    key metrics suitable for the specified audience.

    Signature inputs  : data_summary, target_audience
    Signature outputs : executive_summary, key_metrics
    """

    def __init__(self, client):
        sig = Signature(
            description = (
                "Read a dataset summary and generate a concise executive "
                "summary plus a list of the 5 most important metrics"
            ),
            inputs  = ["data_summary", "target_audience"],
            outputs = ["executive_summary", "key_metrics"]
        )
        self.module = Predict(sig, client)

    def run(self, data_summary, target_audience="business analyst"):
        """
        Parameters:
            data_summary    : string, output of summarize_dataframe()
            target_audience : string, who will read this report
        """
        return self.module(
            data_summary    = data_summary,
            target_audience = target_audience
        )


class InsightExtractor:
    """
    DSPy ChainOfThought module that extracts actionable insights
    and concrete recommendations from a dataset summary given
    a specific analysis goal.

    Signature inputs  : data_summary, analysis_goal
    Signature outputs : insights, recommendations
    """

    def __init__(self, client):
        sig = Signature(
            description = (
                "Extract 3 numbered actionable insights and 3 numbered "
                "recommendations from a dataset analysis"
            ),
            inputs  = ["data_summary", "analysis_goal"],
            outputs = ["insights", "recommendations"]
        )
        self.module = ChainOfThought(sig, client)

    def run(self, data_summary, analysis_goal):
        """
        Parameters:
            data_summary  : string, output of summarize_dataframe()
            analysis_goal : string, what the analysis is trying to achieve
        """
        return self.module(
            data_summary  = data_summary,
            analysis_goal = analysis_goal
        )


class ReportSectionWriter:
    """
    DSPy Predict module that writes a single section of a formal
    data analysis report given the section type and dataset summary.

    Signature inputs  : data_summary, section_type, word_limit
    Signature outputs : report_section
    """

    def __init__(self, client):
        sig = Signature(
            description = (
                "Write one section of a professional data analysis report "
                "using statistics from the dataset summary provided"
            ),
            inputs  = ["data_summary", "section_type", "word_limit"],
            outputs = ["report_section"]
        )
        self.module = Predict(sig, client)

    def write(self, data_summary, section_type, word_limit=150):
        """
        Parameters:
            data_summary : string, output of summarize_dataframe()
            section_type : string, e.g. "Introduction", "Key Findings"
            word_limit   : int, approximate maximum word count
        """
        return self.module(
            data_summary = data_summary,
            section_type = section_type,
            word_limit   = str(word_limit) + " words maximum"
        )


class AnomalyNarrator:
    """
    DSPy ChainOfThought module that reads dataset statistics and
    identifies any unusual values, outliers, or suspicious patterns
    that a data analyst should investigate.

    Signature inputs  : data_summary
    Signature outputs : anomalies_found, severity_assessment
    """

    def __init__(self, client):
        sig = Signature(
            description = (
                "Identify anomalies, outliers, and data quality issues "
                "in the provided dataset statistics"
            ),
            inputs  = ["data_summary"],
            outputs = ["anomalies_found", "severity_assessment"]
        )
        self.module = ChainOfThought(sig, client)

    def analyze(self, data_summary):
        """
        Parameters:
            data_summary : string, output of summarize_dataframe()
        """
        return self.module(data_summary=data_summary)


# ============================================================
# FULL DSPY PIPELINE
# ============================================================

class DSPyPipeline:
    """
    Chains multiple DSPy-style modules together to produce a
    complete data analysis output from raw dataset statistics.
    The pipeline runs four modules in sequence:
      1. DataSummarizer     - executive summary and key metrics
      2. InsightExtractor   - insights and recommendations
      3. ReportSectionWriter - three report sections
      4. AnomalyNarrator    - anomaly and data quality report

    Each module uses either Predict or ChainOfThought internally
    and all communicate through the OllamaClient.
    """

    def __init__(self):
        self.client  = OllamaClient()
        self.df      = load_titanic()
        self.summary = summarize_dataframe(self.df)

        self.summarizer  = DataSummarizer(self.client)
        self.extractor   = InsightExtractor(self.client)
        self.writer      = ReportSectionWriter(self.client)
        self.narrator    = AnomalyNarrator(self.client)

        print("DSPy Pipeline initialized.")
        if DSPY_AVAILABLE:
            print("Real dspy-ai library is available for advanced use.")
        else:
            print("Using DSPy-style mock modules (dspy-ai not installed).")
            print("Install with: pip install dspy-ai")

    def run(self):
        """Runs all four modules on the Titanic dataset in sequence."""

        print("\n-- Module 1: Data Summarization --")
        summary_output = self.summarizer.run(
            self.summary, target_audience="business executive"
        )
        print(summary_output.get("executive_summary", "")[:350])

        print("\n-- Module 2: Insight Extraction --")
        insight_output = self.extractor.run(
            self.summary,
            analysis_goal="identify the key factors affecting passenger survival"
        )
        print(insight_output.get("reasoning", "")[:350])

        print("\n-- Module 3: Report Sections --")
        for section in ["Introduction", "Key Findings", "Recommendations"]:
            section_output = self.writer.write(
                self.summary, section_type=section, word_limit=120
            )
            print("\nSection: " + section)
            print(section_output.get("report_section", "")[:250])

        print("\n-- Module 4: Anomaly Detection --")
        anomaly_output = self.narrator.analyze(self.summary)
        print(anomaly_output.get("reasoning", "")[:350])

        return {
            "summary"  : summary_output,
            "insights" : insight_output,
            "anomalies": anomaly_output
        }

    def demonstrate_few_shot(self):
        """
        Shows how adding labeled examples to a prompt changes
        the style and quality of the LLM output. Runs the same
        question with zero examples and then with two examples
        to show the difference in specificity and format.
        """
        examples = [
            {
                "input" : "survival_rate=38.4%, female_rate=74.2%, male_rate=18.9%",
                "output": "Gender was the dominant survival predictor. Female passengers survived at 74.2%, nearly four times the male rate of 18.9%, a gap of 55.3 percentage points."
            },
            {
                "input" : "class_1=63.0%, class_2=47.3%, class_3=24.2%",
                "output": "Passenger class strongly correlated with survival. First-class passengers survived at 63.0% versus 24.2% for third class, a 38.8 point gap suggesting evacuation priority was tied to socioeconomic status."
            }
        ]

        examples_block = ""
        for i, ex in enumerate(examples):
            examples_block += (
                "Example " + str(i + 1) + ":\n"
                "Input  : " + ex["input"] + "\n"
                "Output : " + ex["output"] + "\n\n"
            )

        zero_shot_prompt = (
            "Generate a concise analytical insight from this data:\n"
            "Input: " + self.summary[:300] + "\nOutput:"
        )

        few_shot_prompt = (
            "Generate a concise analytical insight from this data. "
            "Follow the style of the examples below.\n\n"
            + examples_block
            + "Now generate an insight for this new data:\n"
            "Input: " + self.summary[:300] + "\nOutput:"
        )

        print("\n-- Few-Shot vs Zero-Shot Comparison --")

        print("\n[Zero-shot (no examples)]")
        zero_shot_response = self.client.generate(MODEL, zero_shot_prompt)
        print(zero_shot_response[:350])

        print("\n[Few-shot (2 examples provided)]")
        few_shot_response = self.client.generate(MODEL, few_shot_prompt)
        print(few_shot_response[:350])

        return {
            "zero_shot": zero_shot_response,
            "few_shot" : few_shot_response
        }


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("DSPY PIPELINE DEMO")
    print("=" * 55)

    pipeline = DSPyPipeline()

    print("\n-- Dataset Summary --")
    print(pipeline.summary)

    pipeline.run()
    pipeline.demonstrate_few_shot()

    print("\n-- DSPy Pipeline demo complete --")


if __name__ == "__main__":
    run_demo()
