"""
tot_dspy_pipeline.py  ·  Day 33
---------------------------------
Tree-of-Thought + DSPy for multi-step statistical reasoning.
Explores 3 branches (parametric / non-parametric / Bayesian),
scores each, picks the most statistically sound answer.

Usage:
    python tot_dspy_pipeline.py
    python tot_dspy_pipeline.py --problem anova
"""

import os, json, argparse
import dspy
from dotenv import load_dotenv

load_dotenv()

PROBLEMS = {
    "ttest": (
        "A/B test: version A 1200 visitors avg session 4.2min sd=1.1; "
        "version B 1150 visitors avg session 4.6min sd=1.3. "
        "Is the difference significant? Which test, assumptions, conclusion?"
    ),
    "anova": (
        "Sales from 4 regions: North=82k, South=91k, East=78k, West=95k. "
        "Equal variances. Test if differences are significant and identify which pairs differ."
    ),
    "regression": (
        "House price model: R²=0.71 but residuals show a fan-shape pattern. "
        "Diagnose the problem and recommend the correct fix."
    ),
    "feature_selection": (
        "200 features, 500 samples. Train acc=0.88, test acc=0.61. "
        "Design a principled feature selection strategy using statistical methods."
    ),
}

BRANCHES = [
    "parametric (normality assumed — t-test / ANOVA / linear regression)",
    "non-parametric (no distribution assumption — Mann-Whitney / Kruskal-Wallis / Spearman)",
    "Bayesian (incorporates prior knowledge — credible intervals / posterior inference)",
]


# ── DSPy Signatures ────────────────────────────────────────────────────────

class BranchReason(dspy.Signature):
    """Reason about a DS/stats problem using one specific statistical approach."""
    problem      = dspy.InputField()
    approach     = dspy.InputField(desc="Statistical approach to use for this branch")
    analysis     = dspy.OutputField(desc="Step-by-step analysis using this approach")
    conclusion   = dspy.OutputField(desc="Final answer for this branch")
    confidence   = dspy.OutputField(desc="Confidence score 1-10 and brief justification")


class PickBest(dspy.Signature):
    """Pick the most statistically correct reasoning branch."""
    problem          = dspy.InputField()
    branches_json    = dspy.InputField(desc="All branches as JSON string")
    best_branch_num  = dspy.OutputField(desc="Which branch is best (1, 2, or 3)")
    justification    = dspy.OutputField(desc="Why this branch is statistically correct")
    final_answer     = dspy.OutputField(desc="Synthesised final answer combining best insights")


# ── ToT Module ─────────────────────────────────────────────────────────────

class ToTStatReasoner(dspy.Module):
    def __init__(self):
        super().__init__()
        self.branch_cot = dspy.ChainOfThought(BranchReason)
        self.picker     = dspy.ChainOfThought(PickBest)

    def forward(self, problem: str):
        branches = []
        for i, approach in enumerate(BRANCHES, 1):
            print(f"  exploring branch {i}: {approach[:50]}...")
            r = self.branch_cot(problem=problem, approach=approach)
            branches.append({"num": i, "approach": approach,
                             "analysis": r.analysis,
                             "conclusion": r.conclusion,
                             "confidence": r.confidence})

        result = self.picker(problem=problem, branches_json=json.dumps(branches, indent=2))
        return dspy.Prediction(branches=branches,
                               best=result.best_branch_num,
                               justification=result.justification,
                               final_answer=result.final_answer)


def run_tot(problem_key: str = "ttest") -> dict:
    lm = dspy.OpenAI(model="gpt-4o",
                     api_key=os.getenv("OPENAI_API_KEY"),
                     max_tokens=1024)
    dspy.settings.configure(lm=lm)

    problem = PROBLEMS.get(problem_key, problem_key)
    print(f"\nProblem: {problem_key}")
    print("Running Tree-of-Thought...\n")

    tot    = ToTStatReasoner()
    result = tot(problem=problem)

    print(f"\n── Best Branch: {result.best} ──────────────────────")
    print(f"Justification: {result.justification}")
    print(f"\nFinal Answer:\n{result.final_answer}")

    return {"problem": problem_key,
            "branches_explored": len(result.branches),
            "best_branch": result.best,
            "justification": result.justification,
            "final_answer": result.final_answer}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="ttest", choices=list(PROBLEMS.keys()))
    args = parser.parse_args()
    run_tot(args.problem)
