# ============================================================
# LLAMA PIPELINE
# Day 5 NLP: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Llama3 via Ollama for Titanic dataset NLP analysis
# ============================================================

# Llama3 is Meta's open source large language model released
# in 2024. It performs well on analytical and reasoning tasks
# and is available for free via Ollama. This file uses Llama3
# to read summarized statistics from the Titanic dataset and
# generate natural language insights, predictions, and report
# sections. The key idea is that instead of writing fixed
# templates, we let the LLM produce flexible text from data.
# All data loading uses seaborn's built-in Titanic dataset
# with an inline fallback in case seaborn is not installed.

import pandas as pd
import numpy as np

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

from ollama_setup import OllamaClient

MODEL = "llama3"


# ============================================================
# DATA LOADING
# ============================================================

def load_titanic():
    """
    Loads the Titanic dataset from seaborn if available.
    Falls back to a representative 20-row inline sample
    so the pipeline works without seaborn installed.
    Prints confirmation of which source was used.
    """
    if SEABORN_AVAILABLE:
        df = sns.load_dataset("titanic")
        print("Loaded Titanic dataset from seaborn: " +
              str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns.")
        return df

    data = {
        "survived": [0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
        "pclass"  : [3, 1, 3, 1, 3, 3, 1, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 2, 3, 3],
        "sex"     : ["male", "female", "female", "female", "male", "male",
                     "male", "male", "female", "female", "female", "female",
                     "male", "male", "male", "female", "male", "male", "female", "female"],
        "age"     : [22, 38, 26, 35, 35, None, 54, 2, 27, 14,
                     4, 58, 20, 39, 14, 55, 2, None, 31, None],
        "fare"    : [7.25, 71.28, 7.92, 53.10, 8.05, 8.46, 51.86, 21.07,
                     11.13, 30.07, 16.70, 26.55, 8.05, 31.27, 7.85, 16.00,
                     29.12, 13.00, 18.00, 7.22],
        "embarked": ["S", "C", "S", "S", "S", "Q", "S", "S", "S", "C",
                     "S", "S", "S", "S", "S", "S", "Q", "S", "S", "S"]
    }
    df = pd.DataFrame(data)
    print("Loaded Titanic inline fallback: " +
          str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns.")
    return df


# ============================================================
# DATA SUMMARIZER
# ============================================================

def summarize_dataframe(df):
    """
    Converts a dataframe into a compact text summary suitable
    for sending to an LLM as context. Includes shape, column
    names, missing value counts, numeric statistics, and
    Titanic-specific survival breakdowns where available.

    Returns a multi-line string ready to paste into a prompt.
    """
    lines = []

    lines.append("Dataset shape: " + str(df.shape[0]) +
                 " rows x " + str(df.shape[1]) + " columns")
    lines.append("Columns: " + ", ".join(df.columns.tolist()))
    lines.append("")

    missing = {col: int(df[col].isnull().sum())
               for col in df.columns if df[col].isnull().sum() > 0}
    if missing:
        lines.append("Missing values:")
        for col, count in missing.items():
            pct = round(count / len(df) * 100, 1)
            lines.append("  " + col + ": " + str(count) +
                         " missing (" + str(pct) + "%)")
        lines.append("")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        lines.append("Numeric statistics:")
        for col in num_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                lines.append(
                    "  " + col + ":"
                    " mean=" + str(round(col_data.mean(), 3)) +
                    ", median=" + str(round(col_data.median(), 3)) +
                    ", std=" + str(round(col_data.std(), 3)) +
                    ", min=" + str(round(col_data.min(), 3)) +
                    ", max=" + str(round(col_data.max(), 3))
                )
        lines.append("")

    if "survived" in df.columns:
        overall = round(df["survived"].mean() * 100, 1)
        lines.append("Survival statistics:")
        lines.append("  Overall survival rate: " + str(overall) + "%")

        if "sex" in df.columns:
            for gender in ["male", "female"]:
                subset = df[df["sex"] == gender]["survived"]
                if len(subset) > 0:
                    rate = round(subset.mean() * 100, 1)
                    lines.append("  " + gender.capitalize() +
                                 " survival rate: " + str(rate) + "%")

        if "pclass" in df.columns:
            for cls in [1, 2, 3]:
                subset = df[df["pclass"] == cls]["survived"]
                if len(subset) > 0:
                    rate = round(subset.mean() * 100, 1)
                    lines.append("  Class " + str(cls) +
                                 " survival rate: " + str(rate) + "%")

    return "\n".join(lines)


# ============================================================
# LLAMA PIPELINE CLASS
# ============================================================

class LlamaPipeline:
    """
    Uses Llama3 via Ollama to perform natural language data
    analysis on the Titanic dataset. Each method sends a
    different type of prompt to Llama3 and returns the
    generated text response. The client falls back to mock
    mode automatically if Ollama is not running.
    """

    def __init__(self):
        self.client  = OllamaClient()
        self.df      = load_titanic()
        self.summary = summarize_dataframe(self.df)
        print("Llama pipeline initialized with " +
              str(len(self.df)) + " passenger records.")

    def analyze_dataset(self):
        """
        Sends the full data summary to Llama3 and asks for
        three key business insights formatted as numbered points.
        """
        prompt = (
            "You are a senior data analyst writing a report. "
            "Read the following dataset summary and provide exactly "
            "3 numbered key insights that a business stakeholder "
            "would find valuable. Be specific and reference the numbers.\n\n"
            "Dataset Summary:\n" + self.summary + "\n\n"
            "Key Insights:"
        )
        print("\n-- Llama3 Dataset Analysis --")
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def explain_survival_factors(self):
        """
        Asks Llama3 to explain the top three factors affecting
        passenger survival using the statistics in the summary.
        """
        prompt = (
            "Based on the following Titanic dataset summary, explain "
            "the top 3 factors that most strongly influenced whether a "
            "passenger survived. Reference specific percentages and "
            "statistics from the summary in your explanation.\n\n"
            "Dataset Summary:\n" + self.summary + "\n\n"
            "Top 3 Survival Factors:"
        )
        print("\n-- Llama3 Survival Factor Explanation --")
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def predict_passenger_survival(self, pclass, sex, age, fare):
        """
        Sends a single passenger's details to Llama3 and asks
        for a survival prediction with justification based on
        the patterns observed in the dataset summary.

        Parameters:
            pclass : int, passenger class (1, 2, or 3)
            sex    : str, "male" or "female"
            age    : float, passenger age in years
            fare   : float, ticket fare paid
        """
        passenger_str = (
            "Class: " + str(pclass) + "\n"
            "Sex: " + sex + "\n"
            "Age: " + str(age) + " years\n"
            "Fare paid: $" + str(fare)
        )
        prompt = (
            "Using the survival patterns from the Titanic dataset summary "
            "below, predict whether the following passenger survived. "
            "Give a prediction of Survived or Did Not Survive and explain "
            "your reasoning in 2-3 sentences.\n\n"
            "Historical Data Summary:\n" + self.summary + "\n\n"
            "Passenger Details:\n" + passenger_str + "\n\n"
            "Prediction and Reasoning:"
        )
        print("\n-- Llama3 Passenger Prediction --")
        print("Passenger: Class=" + str(pclass) + ", " + sex +
              ", Age=" + str(age) + ", Fare=$" + str(fare))
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def generate_report_section(self, section_title):
        """
        Generates one section of a professional data analysis
        report. The section title controls the focus of the
        generated text. Keeps output to 3-4 sentences.

        Parameters:
            section_title : str, e.g. "Executive Summary" or
                            "Key Findings" or "Recommendations"
        """
        prompt = (
            "Write a professional data analysis report section "
            "titled '" + section_title + "' based on the Titanic "
            "dataset summary below. Write 3 to 4 sentences in a "
            "formal analytical tone. Reference specific statistics.\n\n"
            "Dataset Summary:\n" + self.summary + "\n\n"
            + section_title + ":"
        )
        print("\n-- Llama3 Report Section: " + section_title + " --")
        response = self.client.generate(MODEL, prompt)
        print(response)
        return response

    def compare_passenger_classes(self):
        """
        Computes per-class survival statistics, formats them
        as a table string, then sends to Llama3 asking for
        an interpretation of socioeconomic factors in survival.
        """
        if "pclass" in self.df.columns and "survived" in self.df.columns:
            stats = self.df.groupby("pclass")["survived"].agg(
                count="count",
                survivors="sum",
                survival_rate="mean"
            )
            stats["survival_rate"] = (stats["survival_rate"] * 100).round(1)
            stats_str = stats.to_string()
        else:
            stats_str = "Passenger class data not available in this sample."

        prompt = (
            "The following table shows Titanic passenger survival "
            "broken down by passenger class (1=first, 2=second, "
            "3=third). Interpret what these numbers reveal about "
            "the role of socioeconomic status in survival outcomes. "
            "Write 3 sentences.\n\n"
            "Class Survival Table:\n" + stats_str + "\n\n"
            "Interpretation:"
        )
        print("\n-- Llama3 Class Comparison Analysis --")
        print("Class statistics:\n" + stats_str)
        response = self.client.generate(MODEL, prompt)
        print("\nLlama3 interpretation: " + response)
        return response


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("LLAMA PIPELINE DEMO")
    print("=" * 55)

    pipeline = LlamaPipeline()

    print("\n-- Dataset Preview (first 5 rows) --")
    preview_cols = [c for c in ["survived", "pclass", "sex", "age", "fare"]
                    if c in pipeline.df.columns]
    print(pipeline.df[preview_cols].head(5).to_string())

    print("\n-- Dataset Summary --")
    print(pipeline.summary)

    pipeline.analyze_dataset()
    pipeline.explain_survival_factors()
    pipeline.compare_passenger_classes()

    pipeline.predict_passenger_survival(pclass=1, sex="female", age=28, fare=100.0)
    pipeline.predict_passenger_survival(pclass=3, sex="male",   age=25, fare=7.25)
    pipeline.predict_passenger_survival(pclass=2, sex="female", age=14, fare=30.07)

    pipeline.generate_report_section("Executive Summary")
    pipeline.generate_report_section("Key Findings")
    pipeline.generate_report_section("Recommendations")

    print("\n-- Llama Pipeline demo complete --")


if __name__ == "__main__":
    run_demo()
