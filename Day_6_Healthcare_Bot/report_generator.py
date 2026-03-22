# ============================================================
# REPORT GENERATOR
# Day 6 Healthcare Bot: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: Automated data analysis report generation with LLMs
# ============================================================

# Generating a professional data analysis report manually takes
# hours. This file automates that process by computing all
# statistics directly from the dataframe for accuracy and then
# using an LLM to write the narrative sections. The report
# follows the structure used by data science teams: title,
# executive summary, methodology, key findings, survival
# breakdown table, recommendations, and conclusion. Each
# section is generated independently so individual sections
# can be regenerated without rebuilding the whole report.
# The final output is assembled as plain text and saved to
# a file. All numeric values come from computed dataframe
# statistics not from the LLM to prevent hallucination.

import os
from datetime import datetime
import pandas as pd
import numpy as np

from data_loader import load_titanic, summarize_dataframe
from openai_client import OpenAIClient, ANALYST_SYSTEM_PROMPT


# ============================================================
# STATISTICS COMPUTER
# ============================================================

def compute_report_statistics(df):
    """
    Computes all statistics needed for the report directly from
    the dataframe. Returns a dict of values passed to the LLM
    as factual context. This ensures all numbers in the report
    are accurate and not hallucinated by the model.

    Parameters:
        df : pandas DataFrame with Titanic passenger data

    Returns:
        dict with string keys and numeric or string values
    """
    stats = {}

    stats["total_records"] = len(df)
    stats["total_columns"] = len(df.columns)
    stats["columns"]       = ", ".join(df.columns.tolist())

    if "survived" in df.columns:
        stats["overall_survival_rate"] = round(df["survived"].mean() * 100, 1)
        stats["survivors_count"]       = int(df["survived"].sum())
        stats["non_survivors_count"]   = int((df["survived"] == 0).sum())

    if "sex" in df.columns and "survived" in df.columns:
        for gender in ["female", "male"]:
            subset = df[df["sex"] == gender]
            if len(subset) > 0:
                stats[gender + "_survival_rate"] = round(subset["survived"].mean() * 100, 1)
                stats[gender + "_count"]         = len(subset)

    if "pclass" in df.columns and "survived" in df.columns:
        for cls in [1, 2, 3]:
            subset = df[df["pclass"] == cls]
            if len(subset) > 0:
                stats["class" + str(cls) + "_survival_rate"] = round(
                    subset["survived"].mean() * 100, 1
                )
                stats["class" + str(cls) + "_count"] = len(subset)

    if "age" in df.columns:
        age_data = df["age"].dropna()
        if len(age_data) > 0:
            stats["age_mean"]    = round(age_data.mean(), 1)
            stats["age_median"]  = round(age_data.median(), 1)
            stats["age_missing"] = int(df["age"].isnull().sum())

    if "fare" in df.columns:
        fare_data = df["fare"].dropna()
        if len(fare_data) > 0:
            stats["fare_mean"]   = round(fare_data.mean(), 2)
            stats["fare_median"] = round(fare_data.median(), 2)
            stats["fare_max"]    = round(fare_data.max(), 2)

    missing_info = {}
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        if missing > 0:
            missing_info[col] = missing
    stats["missing_values"] = missing_info

    return stats


def stats_to_context_string(stats):
    """
    Converts the statistics dict to a formatted multi-line
    string suitable for including in LLM prompts as context.
    """
    lines = []
    for key, value in stats.items():
        if isinstance(value, dict):
            lines.append(key + ":")
            for k, v in value.items():
                lines.append("  " + str(k) + ": " + str(v))
        else:
            lines.append(key + ": " + str(value))
    return "\n".join(lines)


# ============================================================
# REPORT SECTION CLASS
# ============================================================

class ReportSection:
    """
    Represents one section of the data analysis report.
    Stores the title, content text, and generation source.
    """

    def __init__(self, title, content, generated_by="computed"):
        self.title        = title
        self.content      = content
        self.generated_by = generated_by
        self.timestamp    = datetime.now().isoformat()

    def to_text(self):
        """Formats the section as plain text with divider."""
        divider = "=" * len(self.title)
        return "\n" + self.title.upper() + "\n" + divider + "\n\n" + self.content + "\n"


# ============================================================
# REPORT GENERATOR CLASS
# ============================================================

class ReportGenerator:
    """
    Generates a complete structured data analysis report for
    the Titanic dataset. Statistics are computed from the
    dataframe and the LLM writes the narrative sections.
    Sections are generated independently and assembled at end.

    Parameters:
        client      : OpenAIClient or ClaudeClient instance
        client_name : string label e.g. GPT or Claude
    """

    def __init__(self, client, client_name="GPT"):
        self.client      = client
        self.client_name = client_name
        self.df          = load_titanic()
        self.stats       = compute_report_statistics(self.df)
        self.context     = stats_to_context_string(self.stats)
        self.summary     = summarize_dataframe(self.df)
        self.sections    = []
        print("Report Generator ready. Client: " + client_name)

    def _generate(self, prompt, max_tokens=300):
        """Calls the LLM and returns the response string."""
        return self.client.complete(
            prompt,
            system_prompt=ANALYST_SYSTEM_PROMPT,
            max_tokens=max_tokens
        )

    def generate_title_section(self):
        """Creates the report title and metadata section."""
        content = (
            "Report Title  : Titanic Passenger Survival Analysis\n"
            "Dataset       : Titanic Passenger Records\n"
            "Total Records : " + str(self.stats.get("total_records", "N/A")) + "\n"
            "Generated by  : " + self.client_name + " Analytics Pipeline\n"
            "Generated on  : " + datetime.now().strftime("%Y-%m-%d %H:%M") + "\n"
            "Author        : Sheshikala"
        )
        section = ReportSection("Report Information", content, "computed")
        self.sections.append(section)
        return section

    def generate_executive_summary(self):
        """Generates a 2-3 sentence executive summary using the LLM."""
        prompt = (
            "Write a 2-3 sentence executive summary for a data analysis "
            "report on Titanic passenger survival. Reference specific "
            "statistics. Use this data:\n\n" + self.context
        )
        content = self._generate(prompt, max_tokens=200)
        section = ReportSection("Executive Summary", content, self.client_name)
        self.sections.append(section)
        return section

    def generate_methodology(self):
        """Creates the methodology section describing the approach."""
        content = (
            "Data Source    : Titanic passenger records (seaborn dataset)\n"
            "Total Records  : " + str(self.stats.get("total_records", "N/A")) + " passengers\n"
            "Features       : " + self.stats.get("columns", "N/A") + "\n"
            "Target Variable: survived (binary: 0=No, 1=Yes)\n"
            "Analysis Type  : Descriptive statistics and group comparison\n"
            "Tools Used     : Python, Pandas, NumPy, OpenAI/Claude API\n"
            "Approach       : Statistics computed from raw dataframe, LLM used\n"
            "                 only for narrative text generation"
        )
        section = ReportSection("Methodology", content, "computed")
        self.sections.append(section)
        return section

    def generate_key_findings(self):
        """Generates 5 numbered key findings based on computed statistics."""
        prompt = (
            "Based on these Titanic dataset statistics, write exactly "
            "5 numbered key findings. Each finding must reference a "
            "specific number from the data.\n\n" + self.context
        )
        content = self._generate(prompt, max_tokens=400)
        section = ReportSection("Key Findings", content, self.client_name)
        self.sections.append(section)
        return section

    def generate_survival_breakdown(self):
        """Creates a computed survival breakdown table by group."""
        lines = ["Survival rates computed directly from dataset:\n"]

        if "female_survival_rate" in self.stats:
            lines.append("By Gender:")
            lines.append("  Female : " + str(self.stats["female_survival_rate"]) +
                         "% (" + str(self.stats.get("female_count", "N/A")) + " passengers)")
            lines.append("  Male   : " + str(self.stats.get("male_survival_rate", "N/A")) +
                         "% (" + str(self.stats.get("male_count", "N/A")) + " passengers)")

        if "class1_survival_rate" in self.stats:
            lines.append("")
            lines.append("By Passenger Class:")
            for cls in [1, 2, 3]:
                rate  = self.stats.get("class" + str(cls) + "_survival_rate", "N/A")
                count = self.stats.get("class" + str(cls) + "_count", "N/A")
                lines.append("  Class " + str(cls) + " : " + str(rate) +
                             "% (" + str(count) + " passengers)")

        lines.append("")
        lines.append("Overall : " +
                     str(self.stats.get("overall_survival_rate", "N/A")) +
                     "% (" + str(self.stats.get("total_records", "N/A")) + " passengers)")

        section = ReportSection("Survival Breakdown", "\n".join(lines), "computed")
        self.sections.append(section)
        return section

    def generate_recommendations(self):
        """Generates 3 actionable recommendations from data patterns."""
        prompt = (
            "Based on the Titanic survival analysis data below, write "
            "exactly 3 numbered recommendations for improving passenger "
            "safety in maritime transportation. Each recommendation must "
            "reference a specific statistic from the data.\n\n" + self.context
        )
        content = self._generate(prompt, max_tokens=300)
        section = ReportSection("Recommendations", content, self.client_name)
        self.sections.append(section)
        return section

    def generate_conclusion(self):
        """Generates a 2-sentence conclusion for the report."""
        prompt = (
            "Write a 2-sentence conclusion for a Titanic survival "
            "analysis report. Summarize the most important finding "
            "and its implication. Data:\n\n" + self.context[:500]
        )
        content = self._generate(prompt, max_tokens=150)
        section = ReportSection("Conclusion", content, self.client_name)
        self.sections.append(section)
        return section

    def build_full_report(self):
        """
        Generates all report sections in order and assembles
        them into a complete formatted report string.

        Returns:
            string containing the complete formatted report
        """
        print("\nGenerating full report...")
        self.sections = []

        self.generate_title_section()
        self.generate_executive_summary()
        self.generate_methodology()
        self.generate_key_findings()
        self.generate_survival_breakdown()
        self.generate_recommendations()
        self.generate_conclusion()

        report_lines = [
            "=" * 55,
            "TITANIC PASSENGER SURVIVAL ANALYSIS REPORT",
            "=" * 55,
        ]
        for section in self.sections:
            report_lines.append(section.to_text())

        return "\n".join(report_lines)

    def save_report(self, filepath="titanic_report.txt"):
        """
        Generates and saves the full report to a text file.

        Parameters:
            filepath : string, output file path

        Returns:
            filepath string
        """
        report_text = self.build_full_report()
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(report_text)
        print("Report saved to: " + filepath)
        return filepath


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("REPORT GENERATOR DEMO")
    print("=" * 55)

    client    = OpenAIClient()
    generator = ReportGenerator(client, client_name="GPT")

    print("\n-- Computed Statistics --")
    for key, val in list(generator.stats.items())[:12]:
        print("  " + key + ": " + str(val))

    print("\n-- Individual Sections --")
    generator.generate_title_section().to_text()
    print(generator.sections[-1].to_text())

    generator.generate_executive_summary()
    print(generator.sections[-1].to_text())

    generator.generate_survival_breakdown()
    print(generator.sections[-1].to_text())

    generator.generate_key_findings()
    print(generator.sections[-1].to_text())

    generator.generate_recommendations()
    print(generator.sections[-1].to_text())

    generator.generate_conclusion()
    print(generator.sections[-1].to_text())

    print("\n-- Full Report Preview --")
    full_report = generator.build_full_report()
    print(full_report[:600] + "\n... [truncated] ...")

    report_path = generator.save_report("titanic_report.txt")
    print("Saved to: " + report_path)

    if os.path.exists("titanic_report.txt"):
        os.remove("titanic_report.txt")

    print("\n-- Report Generator demo complete --")


if __name__ == "__main__":
    run_demo()
