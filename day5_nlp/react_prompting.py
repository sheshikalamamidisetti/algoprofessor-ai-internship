# ============================================================
# REACT PROMPTING
# Day 5 NLP: DataAssist Analytics Agent
# Author: Sheshikala
# Topic: ReAct agent with data analysis tools for Titanic data
# ============================================================

# ReAct stands for Reasoning and Acting. It is a prompting
# pattern introduced in a 2022 paper from Princeton and Google
# where the LLM alternates between three phases in a loop:
#   Thought     - the model reasons about what to do next
#   Action      - the model decides which tool to call
#   Observation - the tool runs and returns a result
# The loop continues until the model emits FINISH with a
# final answer. This is more powerful than standard prompting
# because the model can run real calculations at each step
# rather than guessing from memory. This file defines five
# analysis tools that operate on the Titanic dataframe and
# a ReActAgent class that interprets the model output and
# dispatches to the correct tool automatically.

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
# ANALYSIS TOOLS
# ============================================================
# Each tool function takes the dataframe as its first argument
# and returns a plain string result. The ReActAgent calls
# these functions based on the Action line in the LLM response.

def tool_survival_rate(df, group_col=None, group_val=None):
    """
    Calculates the survival rate as a percentage.
    If group_col and group_val are provided, calculates
    the rate for that specific subgroup only.
    If no group is specified, returns the overall rate.

    Examples:
        tool_survival_rate(df)
        tool_survival_rate(df, "sex", "female")
        tool_survival_rate(df, "pclass", 1)
    """
    if "survived" not in df.columns:
        return "Survival column not found in dataset."

    if group_col and group_val is not None and group_col in df.columns:
        subset = df[df[group_col] == group_val]
        if len(subset) == 0:
            return "No records found for " + str(group_col) + "=" + str(group_val)
        rate  = round(subset["survived"].mean() * 100, 1)
        count = len(subset)
        return (str(group_col) + "=" + str(group_val) +
                ": survival rate=" + str(rate) + "% (" +
                str(count) + " passengers)")

    rate  = round(df["survived"].mean() * 100, 1)
    count = len(df)
    return "Overall survival rate=" + str(rate) + "% (" + str(count) + " passengers)"


def tool_descriptive_stats(df, column):
    """
    Returns mean, median, standard deviation, minimum, and
    maximum for a numeric column in the dataframe.

    Parameters:
        df     : pandas DataFrame
        column : column name string
    """
    if column not in df.columns:
        return "Column '" + column + "' not found. Available: " + ", ".join(df.columns.tolist())

    col_data = df[column].dropna()
    if len(col_data) == 0:
        return "No non-null values found in column: " + column

    return (
        column + " descriptive statistics:"
        " mean=" + str(round(col_data.mean(), 3)) +
        ", median=" + str(round(col_data.median(), 3)) +
        ", std=" + str(round(col_data.std(), 3)) +
        ", min=" + str(round(col_data.min(), 3)) +
        ", max=" + str(round(col_data.max(), 3)) +
        ", count=" + str(len(col_data))
    )


def tool_value_counts(df, column, top_n=5):
    """
    Returns the frequency distribution of values in a
    categorical or low-cardinality column as a formatted string.

    Parameters:
        df     : pandas DataFrame
        column : column name string
        top_n  : maximum number of values to return
    """
    if column not in df.columns:
        return "Column '" + column + "' not found."

    counts = df[column].value_counts().head(top_n)
    total  = len(df)
    lines  = [column + " value distribution:"]
    for value, count in counts.items():
        pct = round(count / total * 100, 1)
        lines.append(
            "  " + str(value) + ": " + str(count) +
            " records (" + str(pct) + "%)"
        )
    return "\n".join(lines)


def tool_correlation(df, col_a, col_b):
    """
    Computes the Pearson correlation coefficient between two
    numeric columns. Returns the value with a plain English
    interpretation of the strength and direction.

    Parameters:
        df    : pandas DataFrame
        col_a : first column name
        col_b : second column name
    """
    for col in [col_a, col_b]:
        if col not in df.columns:
            return "Column '" + col + "' not found in dataset."

    pair    = df[[col_a, col_b]].dropna()
    if len(pair) < 2:
        return "Not enough data to compute correlation between " + col_a + " and " + col_b

    corr = round(pair.corr().iloc[0, 1], 4)

    if abs(corr) >= 0.7:
        strength = "strong"
    elif abs(corr) >= 0.4:
        strength = "moderate"
    else:
        strength = "weak"

    direction = "positive" if corr > 0 else "negative"

    return (
        col_a + " vs " + col_b + ":"
        " correlation=" + str(corr) +
        " (" + strength + " " + direction + " relationship,"
        " based on " + str(len(pair)) + " complete records)"
    )


def tool_group_comparison(df, group_col, target_col):
    """
    Groups the dataframe by group_col and computes the mean
    and count of target_col for each group. Returns a formatted
    table string showing the comparison.

    Parameters:
        df         : pandas DataFrame
        group_col  : column to group by e.g. "sex" or "pclass"
        target_col : column to aggregate e.g. "survived" or "fare"
    """
    for col in [group_col, target_col]:
        if col not in df.columns:
            return "Column '" + col + "' not found."

    stats = df.groupby(group_col)[target_col].agg(
        count="count",
        mean="mean"
    )
    stats["mean"] = stats["mean"].round(3)
    return group_col + " vs " + target_col + ":\n" + stats.to_string()


# ============================================================
# TOOL REGISTRY
# ============================================================

AVAILABLE_TOOLS = {
    "survival_rate"    : tool_survival_rate,
    "descriptive_stats": tool_descriptive_stats,
    "value_counts"     : tool_value_counts,
    "correlation"      : tool_correlation,
    "group_comparison" : tool_group_comparison,
}

TOOL_DESCRIPTIONS = (
    "Available tools:\n"
    "  survival_rate(group_col, group_val) - get survival rate overall or for a subgroup\n"
    "  descriptive_stats(column) - get mean, median, std, min, max for a numeric column\n"
    "  value_counts(column) - get frequency distribution of a categorical column\n"
    "  correlation(col_a, col_b) - get correlation coefficient between two numeric columns\n"
    "  group_comparison(group_col, target_col) - compare a metric across groups\n"
    "  FINISH - emit this when the final answer is ready\n"
)


# ============================================================
# REACT AGENT CLASS
# ============================================================

class ReActAgent:
    """
    ReAct agent for multi-step Titanic data analysis.

    The agent sends a structured prompt to Llama3 asking it
    to respond in Thought / Action / Observation format. After
    each model response the agent parses the Action line,
    calls the corresponding tool with the Titanic dataframe,
    and appends the Observation back to the prompt history.
    The loop repeats until FINISH is detected or max_steps
    is reached. This approach lets the model use real data
    calculations at each reasoning step.
    """

    def __init__(self):
        self.client  = OllamaClient()
        self.df      = load_titanic()
        self.summary = summarize_dataframe(self.df)
        print("ReAct Agent initialized with " +
              str(len(self.df)) + " passenger records.")

    def _build_initial_prompt(self, question, history_lines):
        """Constructs the full ReAct prompt including history."""
        history_str = "\n".join(history_lines) if history_lines else "None yet."
        return (
            "You are a data analysis agent. Answer the question by "
            "reasoning and calling tools step by step.\n\n"
            "Dataset context:\n" + self.summary + "\n\n"
            + TOOL_DESCRIPTIONS + "\n"
            "Response format:\n"
            "Thought: [your reasoning about what to do next]\n"
            "Action: [tool_name(argument1, argument2)] or FINISH: [final answer]\n\n"
            "Previous steps:\n" + history_str + "\n\n"
            "Question: " + question + "\n\n"
            "Thought:"
        )

    def _parse_action_line(self, response):
        """
        Extracts the Action line from the model response text.
        Returns the action string or empty string if not found.
        """
        for line in response.strip().split("\n"):
            stripped = line.strip()
            if stripped.startswith("Action:"):
                return stripped[len("Action:"):].strip()
            if stripped.startswith("FINISH"):
                return stripped
        return ""

    def _execute_action(self, action_str):
        """
        Parses the action string and calls the appropriate tool.
        Uses keyword matching to identify the tool and arguments.
        Returns the tool result as a string, or None if FINISH.
        """
        if not action_str or "FINISH" in action_str:
            return None

        action_lower = action_str.lower()

        try:
            if "survival_rate" in action_lower:
                if "female" in action_lower:
                    return tool_survival_rate(self.df, "sex", "female")
                elif "male" in action_lower:
                    return tool_survival_rate(self.df, "sex", "male")
                elif "pclass" in action_lower or "class" in action_lower:
                    for cls in [1, 2, 3]:
                        if str(cls) in action_str:
                            return tool_survival_rate(self.df, "pclass", cls)
                return tool_survival_rate(self.df)

            elif "descriptive_stats" in action_lower:
                for col in ["age", "fare", "survived", "pclass"]:
                    if col in action_lower:
                        return tool_descriptive_stats(self.df, col)
                return tool_descriptive_stats(self.df, "age")

            elif "value_counts" in action_lower:
                for col in ["sex", "pclass", "embarked", "survived"]:
                    if col in action_lower:
                        return tool_value_counts(self.df, col)
                return tool_value_counts(self.df, "sex")

            elif "correlation" in action_lower:
                if "age" in action_lower and "fare" in action_lower:
                    return tool_correlation(self.df, "age", "fare")
                elif "pclass" in action_lower and "fare" in action_lower:
                    return tool_correlation(self.df, "pclass", "fare")
                return tool_correlation(self.df, "age", "fare")

            elif "group_comparison" in action_lower:
                if "sex" in action_lower:
                    return tool_group_comparison(self.df, "sex", "survived")
                elif "pclass" in action_lower or "class" in action_lower:
                    return tool_group_comparison(self.df, "pclass", "survived")
                return tool_group_comparison(self.df, "sex", "survived")

            else:
                return "Tool not recognized in action: " + action_str

        except Exception as e:
            return "Tool execution failed: " + str(e)

    def run(self, question, max_steps=4):
        """
        Runs the ReAct loop for a given analytical question.

        Parameters:
            question  : string, the question to answer
            max_steps : int, maximum Thought-Action-Observation cycles

        Returns:
            list of history strings from all steps
        """
        print("\n-- ReAct Agent --")
        print("Question  : " + question)
        print("Max steps : " + str(max_steps))
        print("-" * 50)

        history = []

        for step in range(1, max_steps + 1):
            print("\nStep " + str(step) + ":")
            prompt   = self._build_initial_prompt(question, history)
            response = self.client.generate(MODEL, prompt)

            print(response[:450])

            action = self._parse_action_line(response)

            if not action:
                print("No action found in response. Stopping.")
                break

            if "FINISH" in action:
                print("\nAgent reached final answer.")
                break

            observation = self._execute_action(action)
            if observation is None:
                print("Action resolved to FINISH.")
                break

            print("Observation: " + observation)

            history.append("Step " + str(step) + " Thought+Action:")
            history.append(response[:300])
            history.append("Observation: " + observation)

        return history


# ============================================================
# DEMO
# ============================================================

def run_demo():
    print("=" * 55)
    print("REACT PROMPTING DEMO")
    print("=" * 55)

    agent = ReActAgent()

    questions = [
        "Which gender had a significantly better survival rate and by how much?",
        "Is there a meaningful correlation between passenger age and the fare paid?",
        "Which passenger class had the lowest survival rate and what was it?",
        "What is the distribution of passenger sexes in the dataset?"
    ]

    for question in questions:
        agent.run(question, max_steps=3)

    print("\n-- ReAct Prompting demo complete --")


if __name__ == "__main__":
    run_demo()
