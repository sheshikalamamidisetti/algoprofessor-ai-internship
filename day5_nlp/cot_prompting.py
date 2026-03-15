# ============================================================
# CHAIN OF THOUGHT PROMPTING
# Day 5 NLP: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: CoT prompt engineering for step-by-step data analysis
# ============================================================

# Chain of Thought prompting is a technique introduced in a
# 2022 Google research paper. Instead of asking an LLM to
# give an answer directly, you ask it to reason step by step
# before reaching a conclusion. For data analysis tasks this
# is particularly useful because it forces the model to check
# the data, apply logic, and justify each step rather than
# jumping straight to a guess. This file applies CoT prompting
# to three types of tasks: general dataset analysis, passenger
# survival prediction, and group comparison. It also includes
# a side-by-side comparison showing the difference in output
# quality between direct prompting and CoT prompting on the
# same Titanic analysis question.

import pandas as pd
import numpy as np

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from ollama_setup import OllamaClient
from llama_pipeline import load_titanic, summarize_dataframe

MODEL = "llama3"


# ============================================================
# COT PROMPT BUILDERS
# ============================================================

def build_cot_analysis_prompt(data_summary, question):
    """
    Wraps an analysis question with Chain of Thought instructions.
    Forces the model to work through four numbered steps before
    providing a final answer. This structure improves accuracy
    on quantitative and logical analysis questions.

    Parameters:
        data_summary : string output from summarize_dataframe()
        question     : the analytical question to answer

    Returns:
        complete prompt string ready to send to the LLM
    """
    return (
        "You are a data analyst. Answer the question below using "
        "step-by-step reasoning. Do not skip steps.\n\n"
        "Dataset Summary:\n" + data_summary + "\n\n"
        "Question: " + question + "\n\n"
        "Work through the following steps:\n"
        "Step 1: Identify which data points are relevant to this question\n"
        "Step 2: Apply the appropriate analysis or calculation\n"
        "Step 3: Interpret what the numbers mean in context\n"
        "Step 4: State a clear, specific final answer\n\n"
        "Step 1:"
    )


def build_cot_prediction_prompt(passenger_info, data_summary):
    """
    Builds a CoT prompt for predicting whether a single Titanic
    passenger survived. The model is guided through evaluating
    each feature separately before combining them into a final
    prediction with a confidence level.

    Parameters:
        passenger_info : formatted string describing the passenger
        data_summary   : string output from summarize_dataframe()

    Returns:
        complete prompt string ready to send to the LLM
    """
    return (
        "You are analyzing Titanic passenger survival data. "
        "Use step-by-step reasoning to predict whether the "
        "following passenger survived the disaster.\n\n"
        "Historical Survival Statistics:\n" + data_summary + "\n\n"
        "Passenger to Evaluate:\n" + passenger_info + "\n\n"
        "Reasoning steps:\n"
        "Step 1: Evaluate this passenger's class. How does class correlate with survival?\n"
        "Step 2: Evaluate the passenger's sex. How does sex correlate with survival?\n"
        "Step 3: Evaluate the passenger's age. Does age factor into survival odds?\n"
        "Step 4: Evaluate the fare paid. Does fare level indicate any additional factors?\n"
        "Step 5: Combine all factors. State your final prediction as Survived or "
        "Did Not Survive with a confidence percentage and one-sentence justification.\n\n"
        "Step 1:"
    )


def build_cot_comparison_prompt(group_a, group_b, metric, data_summary):
    """
    Builds a CoT prompt for comparing two passenger groups on a
    specific metric. The model is guided to examine each group
    individually before calculating the difference and explaining
    the underlying cause.

    Parameters:
        group_a      : description of first group e.g. "male passengers"
        group_b      : description of second group e.g. "female passengers"
        metric       : the metric to compare e.g. "survival rate"
        data_summary : string output from summarize_dataframe()

    Returns:
        complete prompt string ready to send to the LLM
    """
    return (
        "Compare " + group_a + " and " + group_b + " on the metric "
        "of " + metric + " using step-by-step reasoning.\n\n"
        "Dataset Summary:\n" + data_summary + "\n\n"
        "Step 1: What is the " + metric + " for " + group_a + "?\n"
        "Step 2: What is the " + metric + " for " + group_b + "?\n"
        "Step 3: Calculate the absolute difference and direction\n"
        "Step 4: Explain what historical or structural factors caused this difference\n"
        "Step 5: State the conclusion in one clear sentence\n\n"
        "Step 1:"
    )


# ============================================================
# COT PIPELINE CLASS
# ============================================================

class CoTPipeline:
    """
    Applies Chain of Thought prompting to Titanic data analysis.
    Each method constructs a structured multi-step prompt and
    sends it to Llama3 via OllamaClient. The class also provides
    an evaluation method that compares direct prompting versus
    CoT prompting on the same question to demonstrate the
    quality improvement CoT provides.
    """

    def __init__(self):
        self.client  = OllamaClient()
        self.df      = load_titanic()
        self.summary = summarize_dataframe(self.df)
        print("CoT Pipeline initialized with " +
              str(len(self.df)) + " passenger records.")

    def analyze_with_cot(self, question):
        """
        Runs Chain of Thought analysis for a given question
        about the Titanic dataset. Prints and returns the
        step-by-step response from the model.

        Parameters:
            question : string, the analytical question to answer
        """
        prompt = build_cot_analysis_prompt(self.summary, question)
        print("\n-- CoT Analysis --")
        print("Question: " + question)
        print("-" * 45)
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def predict_survival_cot(self, pclass, sex, age, fare):
        """
        Uses Chain of Thought reasoning to predict and explain
        survival for a passenger with the given attributes.

        Parameters:
            pclass : int, passenger class 1, 2, or 3
            sex    : str, "male" or "female"
            age    : float, passenger age in years
            fare   : float, ticket fare paid in dollars
        """
        passenger_info = (
            "Passenger class : " + str(pclass) + "\n"
            "Sex             : " + sex + "\n"
            "Age             : " + str(age) + " years\n"
            "Fare paid       : $" + str(fare)
        )
        prompt = build_cot_prediction_prompt(passenger_info, self.summary)
        print("\n-- CoT Survival Prediction --")
        print("Evaluating: Class=" + str(pclass) + ", " +
              sex + ", Age=" + str(age) + ", Fare=$" + str(fare))
        print("-" * 45)
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def compare_groups_cot(self, group_a, group_b, metric):
        """
        Compares two passenger groups on a given metric using
        Chain of Thought reasoning.

        Parameters:
            group_a : str, description of first group
            group_b : str, description of second group
            metric  : str, the metric to compare
        """
        prompt = build_cot_comparison_prompt(
            group_a, group_b, metric, self.summary
        )
        print("\n-- CoT Group Comparison --")
        print("Comparing: " + group_a + " vs " + group_b)
        print("Metric   : " + metric)
        print("-" * 45)
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def evaluate_cot_vs_direct(self, question):
        """
        Sends the same question to the model using both a direct
        prompt and a Chain of Thought prompt. Prints both responses
        side by side so the quality improvement is visible.

        Parameters:
            question : str, the analytical question to test
        """
        direct_prompt = (
            "Answer this question about the Titanic dataset: " +
            question + "\n\nDataset Summary:\n" + self.summary
        )
        cot_prompt = build_cot_analysis_prompt(self.summary, question)

        print("\n-- CoT vs Direct Prompt Comparison --")
        print("Question: " + question)
        print("=" * 55)

        print("\n[Direct Prompt - no reasoning steps]")
        direct_response = self.client.generate(MODEL, direct_prompt)
        print(direct_response[:500])

        print("\n[Chain of Thought Prompt - step-by-step reasoning]")
        cot_response = self.client.generate(MODEL, cot_prompt)
        print(cot_response[:500])

        print("\nDirect response length : " + str(len(direct_response)) + " chars")
        print("CoT response length    : " + str(len(cot_response)) + " chars")

        return {"direct": direct_response, "cot": cot_response}


# ============================================================
# BATCH COT EVALUATION
# ============================================================

def run_batch_cot(pipeline, questions):
    """
    Runs CoT analysis on a list of questions and collects
    all responses. Prints a summary of response lengths
    after all questions are processed.

    Parameters:
        pipeline  : CoTPipeline instance
        questions : list of question strings

    Returns:
        list of dicts with question, response, and length
    """
    print("\n-- Batch CoT Evaluation --")
    print("Processing " + str(len(questions)) + " questions...")

    results = []
    for i, question in enumerate(questions):
        print("\n[" + str(i + 1) + "/" + str(len(questions)) + "] " + question)
        response = pipeline.analyze_with_cot(question)
        results.append({
            "question": question,
            "response": response,
            "length"  : len(response)
        })

    total_length = sum(r["length"] for r in results)
    avg_length   = round(total_length / len(results))
    print("\nBatch complete.")
    print("Total responses  : " + str(len(results)))
    print("Average length   : " + str(avg_length) + " characters")

    return results


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("CHAIN OF THOUGHT PROMPTING DEMO")
    print("=" * 55)

    pipeline = CoTPipeline()

    analytical_questions = [
        "Which passenger group had the highest survival rate and why?",
        "What does the fare distribution tell us about passenger wealth on the Titanic?",
        "How did the port of embarkation relate to passenger class and survival?"
    ]

    for question in analytical_questions:
        pipeline.analyze_with_cot(question)

    print("\n-- CoT Survival Predictions --")
    pipeline.predict_survival_cot(pclass=1, sex="female", age=28, fare=100.00)
    pipeline.predict_survival_cot(pclass=3, sex="male",   age=25, fare=7.25)
    pipeline.predict_survival_cot(pclass=2, sex="female", age=10, fare=23.00)

    print("\n-- CoT Group Comparisons --")
    pipeline.compare_groups_cot(
        "male passengers", "female passengers", "survival rate"
    )
    pipeline.compare_groups_cot(
        "first class passengers", "third class passengers", "survival rate"
    )

    print("\n-- CoT vs Direct Prompt Evaluation --")
    pipeline.evaluate_cot_vs_direct(
        "What was the single most important factor determining "
        "whether a Titanic passenger survived?"
    )

    print("\n-- Chain of Thought Prompting demo complete --")


if __name__ == "__main__":
    run_demo()
