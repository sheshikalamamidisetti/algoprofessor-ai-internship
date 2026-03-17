# ============================================================
# MISTRAL PIPELINE
# Day 5 NLP: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Mistral via Ollama for concise Titanic data analysis
# ============================================================

# Mistral is a 7-billion parameter open source LLM developed
# by Mistral AI in France and released in 2023. Compared to
# Llama3, Mistral is smaller and generally faster to respond
# while still producing high quality analytical text. This
# file uses Mistral to generate concise outputs such as
# quick summaries, bullet-point metrics, and short passenger
# classifications. It also includes a direct comparison
# function that runs the same prompt on both Mistral and
# Llama3 so you can see the difference in style and speed.
# Both models are accessed through the OllamaClient which
# falls back to mock mode if Ollama is not running.

import time
import pandas as pd
import numpy as np

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from ollama_setup import OllamaClient
from llama_pipeline import load_titanic, summarize_dataframe

MODEL = "mistral"


# ============================================================
# AGE GROUP ANALYSIS
# ============================================================

def compute_age_group_stats(df):
    """
    Segments passengers into five age groups and computes
    the survival rate and passenger count for each group.
    Returns a formatted string table suitable for sending
    to an LLM as context in a prompt.

    Age groups:
        child      : 0 to 12 years
        teen       : 13 to 17 years
        adult      : 18 to 35 years
        middle_aged: 36 to 59 years
        senior     : 60 years and above
    """
    df_age = df.dropna(subset=["age"]).copy()

    if len(df_age) == 0:
        return "Age data not available."

    bins   = [0, 12, 17, 35, 59, 120]
    labels = ["child", "teen", "adult", "middle_aged", "senior"]
    df_age["age_group"] = pd.cut(
        df_age["age"], bins=bins, labels=labels, right=True
    )

    if "survived" in df_age.columns:
        stats = df_age.groupby("age_group", observed=True)["survived"].agg(
            count="count",
            survived_count="sum",
            survival_rate="mean"
        )
        stats["survival_rate"] = (stats["survival_rate"] * 100).round(1)
        return stats.to_string()

    counts = df_age["age_group"].value_counts().sort_index()
    return counts.to_string()


# ============================================================
# MISTRAL PIPELINE CLASS
# ============================================================

class MistralPipeline:
    """
    Uses Mistral via Ollama to generate concise natural language
    outputs from Titanic dataset analysis. Mistral is well suited
    for tasks that require short, structured responses such as
    bullet-point summaries, quick metrics extraction, and brief
    passenger assessments. Each method is designed to produce
    output that is shorter and more direct than Llama3 equivalents.
    """

    def __init__(self):
        self.client  = OllamaClient()
        self.df      = load_titanic()
        self.summary = summarize_dataframe(self.df)
        print("Mistral pipeline initialized with " +
              str(len(self.df)) + " passenger records.")

    def quick_summary(self):
        """
        Asks Mistral for a two-sentence dataset summary.
        Useful for generating the opening line of a report
        where brevity matters more than detail.
        """
        prompt = (
            "Summarize the following dataset in exactly 2 sentences. "
            "Be concise and include the most important statistic.\n\n"
            "Dataset Summary:\n" + self.summary + "\n\n"
            "Two-sentence summary:"
        )
        print("\n-- Mistral Quick Summary --")
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def extract_key_metrics(self):
        """
        Asks Mistral to extract the five most important numeric
        metrics from the dataset summary and format them as a
        list explaining why each metric matters for analysis.
        """
        prompt = (
            "From the dataset summary below, identify the 5 most "
            "important numeric metrics that a data analyst should "
            "highlight in a report. Format each as:\n"
            "  Metric name: value - one sentence why it matters\n\n"
            "Dataset Summary:\n" + self.summary + "\n\n"
            "Top 5 Metrics:"
        )
        print("\n-- Mistral Key Metrics Extraction --")
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def generate_recommendations(self):
        """
        Asks Mistral for three specific, data-driven recommendations
        based on the survival patterns in the Titanic dataset. Each
        recommendation should reference actual statistics.
        """
        prompt = (
            "Based on the Titanic survival statistics below, provide "
            "exactly 3 numbered recommendations for improving passenger "
            "safety in maritime travel. Each recommendation must "
            "reference a specific statistic from the data.\n\n"
            "Dataset Summary:\n" + self.summary + "\n\n"
            "Recommendations:"
        )
        print("\n-- Mistral Recommendations --")
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def classify_passengers(self, passengers):
        """
        Sends a list of passenger dictionaries to Mistral and asks
        for a brief survival likelihood assessment for each one.
        Each passenger dict should contain pclass, sex, age, fare.

        Parameters:
            passengers : list of dicts with keys pclass, sex, age, fare
        """
        passenger_str = ""
        for i, p in enumerate(passengers):
            passenger_str += (
                "Passenger " + str(i + 1) + ": "
                "Class=" + str(p["pclass"]) + ", "
                "Sex=" + str(p["sex"]) + ", "
                "Age=" + str(p["age"]) + ", "
                "Fare=$" + str(p["fare"]) + "\n"
            )

        prompt = (
            "Using Titanic survival patterns, assess survival likelihood "
            "for each passenger below. For each one write one sentence "
            "with your assessment and the main reason.\n\n"
            "Historical patterns:\n" + self.summary[:500] + "\n\n"
            "Passengers to assess:\n" + passenger_str + "\n"
            "Assessments:"
        )
        print("\n-- Mistral Passenger Classification --")
        print("Passengers submitted:")
        print(passenger_str)
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def analyze_age_groups(self):
        """
        Computes age group survival statistics, formats them as
        a table, then asks Mistral to interpret what the numbers
        reveal about age as a survival factor on the Titanic.
        """
        age_stats = compute_age_group_stats(self.df)
        prompt = (
            "The table below shows Titanic survival rates broken down "
            "by passenger age group. Interpret what this tells us about "
            "how age affected survival chances. Write 2-3 sentences.\n\n"
            "Age Group Statistics:\n" + age_stats + "\n\n"
            "Interpretation:"
        )
        print("\n-- Mistral Age Group Analysis --")
        print("Age group statistics:\n" + age_stats)
        response = self.client.generate(MODEL, prompt)
        print("\nMistral interpretation: " + response)
        return response

    def compare_with_llama(self, prompt):
        """
        Runs the same prompt on both Mistral and Llama3 and prints
        the response from each model side by side with timing.
        This makes it easy to compare which model is better suited
        for a given type of analytical task.

        Parameters:
            prompt : the prompt string to send to both models
        """
        print("\n-- Model Comparison: Mistral vs Llama3 --")
        print("Prompt: " + prompt[:80] + "...")
        print("-" * 55)

        start            = time.time()
        mistral_response = self.client.generate("mistral", prompt)
        mistral_time     = round(time.time() - start, 3)

        start          = time.time()
        llama_response = self.client.generate("llama3", prompt)
        llama_time     = round(time.time() - start, 3)

        print("\nMistral (" + str(mistral_time) + "s, " +
              str(len(mistral_response)) + " chars):")
        print(mistral_response[:350])

        print("\nLlama3 (" + str(llama_time) + "s, " +
              str(len(llama_response)) + " chars):")
        print(llama_response[:350])

        faster = "Mistral" if mistral_time < llama_time else "Llama3"
        longer = "Mistral" if len(mistral_response) > len(llama_response) else "Llama3"
        print("\nFastest  : " + faster)
        print("Most detailed : " + longer)

        return {
            "mistral": {"response": mistral_response, "time_s": mistral_time},
            "llama3" : {"response": llama_response,   "time_s": llama_time}
        }


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("MISTRAL PIPELINE DEMO")
    print("=" * 55)

    pipeline = MistralPipeline()

    print("\n-- Dataset Summary --")
    print(pipeline.summary)

    pipeline.quick_summary()
    pipeline.extract_key_metrics()
    pipeline.generate_recommendations()

    pipeline.analyze_age_groups()

    sample_passengers = [
        {"pclass": 1, "sex": "female", "age": 29, "fare": 211.34},
        {"pclass": 3, "sex": "male",   "age": 22, "fare": 7.25},
        {"pclass": 2, "sex": "female", "age": 14, "fare": 30.07},
        {"pclass": 3, "sex": "male",   "age": 45, "fare": 8.05},
    ]
    pipeline.classify_passengers(sample_passengers)

    comparison_prompt = (
        "What are the 3 most important insights from the Titanic "
        "dataset for inclusion in a data analytics report?"
    )
    pipeline.compare_with_llama(comparison_prompt)

    print("\n-- Mistral Pipeline demo complete --")


if __name__ == "__main__":
    run_demo()
