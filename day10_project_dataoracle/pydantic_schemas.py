"""
pydantic_schemas.py  ·  Day 34
--------------------------------
All Pydantic v2 schemas for structured LLM outputs.
Every LLM response goes through these schemas before use.

Usage:
    python pydantic_schemas.py   ← runs validation smoke test
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional
from datetime import datetime


class HypothesisTestResult(BaseModel):
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    p_value: float = Field(ge=0, le=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    reject_null: bool
    conclusion: str
    assumptions_met: list[str]
    effect_size: Optional[str] = None

    @field_validator("p_value")
    @classmethod
    def validate_p(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("p_value must be between 0 and 1")
        return round(v, 6)


class ModelMetrics(BaseModel):
    accuracy:   Optional[float] = Field(None, ge=0, le=1)
    precision:  Optional[float] = Field(None, ge=0, le=1)
    recall:     Optional[float] = Field(None, ge=0, le=1)
    f1_score:   Optional[float] = Field(None, ge=0, le=1)
    roc_auc:    Optional[float] = Field(None, ge=0, le=1)
    rmse:       Optional[float] = Field(None, ge=0)
    r_squared:  Optional[float] = None
    cv_mean:    Optional[float] = None
    cv_std:     Optional[float] = None


class MLModelEval(BaseModel):
    model_name: str
    task_type:  Literal["classification", "regression", "clustering"] = "classification"
    dataset:    str
    train:      ModelMetrics
    test:       ModelMetrics
    is_overfitting: bool = False
    diagnosis:      Optional[str] = None
    top_features:   list[str] = []
    recommendations: list[str] = []
    timestamp:  datetime = Field(default_factory=datetime.now)


class LLMBenchmarkEntry(BaseModel):
    llm:       str
    task_id:   str
    category:  Literal["stat_inference", "ml_code", "eda", "reasoning"]
    score_pct: float = Field(ge=0, le=100)
    latency_ms: Optional[float] = None
    tokens:    Optional[int] = None
    strengths:  list[str] = []
    weaknesses: list[str] = []


class DataOracleReport(BaseModel):
    """Master report — the day10_project Milestone deliverable."""
    title:      str = "DataOracle — LLM Insights Predictor Report"
    student:    str = "Sheshikala"
    programme:  str = "IIT Indore AI & Data Science"
    milestone:  str = "M6 — DataOracle Capstone"
    generated_at: datetime = Field(default_factory=datetime.now)

    llm_benchmarks:  list[LLMBenchmarkEntry] = []
    ml_evaluations:  list[MLModelEval] = []
    stat_tests:      list[HypothesisTestResult] = []

    best_llm:        str = ""
    best_llm_reason: str = ""
    best_ml_model:   str = ""

    key_findings:    list[str] = []
    recommendations: list[str] = []
    tools_used:      list[str] = []

    def summary_table(self) -> str:
        from collections import defaultdict
        scores    = defaultdict(list)
        latencies = defaultdict(list)
        for b in self.llm_benchmarks:
            scores[b.llm].append(b.score_pct)
            if b.latency_ms:
                latencies[b.llm].append(b.latency_ms)
        rows = ["| LLM | Avg Score % | Avg Latency ms |",
                "|-----|------------|----------------|"]
        for llm in sorted(scores, key=lambda m: -sum(scores[m])/len(scores[m])):
            avg_s = sum(scores[llm]) / len(scores[llm])
            avg_l = sum(latencies[llm]) / len(latencies[llm]) if latencies[llm] else 0
            rows.append(f"| {llm} | {avg_s:.1f}% | {avg_l:.0f} |")
        return "\n".join(rows)

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            f"**Student:** {self.student} | **Programme:** {self.programme}",
            f"**Milestone:** {self.milestone}",
            f"**Generated:** {self.generated_at.strftime('%B %d, %Y %H:%M')}",
            "", "---", "",
            "## LLM Benchmark Results", "",
            self.summary_table() if self.llm_benchmarks else "_No benchmark data yet._",
            "",
            f"**Best LLM:** {self.best_llm}",
            f"**Reason:** {self.best_llm_reason}",
            "", "---", "", "## Key Findings", "",
        ]
        for i, f in enumerate(self.key_findings, 1):
            lines.append(f"{i}. {f}")
        lines += ["", "---", "", "## Recommendations", ""]
        for r in self.recommendations:
            lines.append(f"- {r}")
        lines += ["", "---", "", f"**Tools:** {', '.join(self.tools_used)}"]
        return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test — validate all schemas
    t = HypothesisTestResult(
        test_name="Independent t-test",
        null_hypothesis="No difference in recovery time",
        alternative_hypothesis="Treatment recovers faster",
        p_value=0.03, reject_null=True,
        conclusion="Reject H0 at alpha=0.05. Treatment is effective.",
        assumptions_met=["normality", "equal variances"],
    )
    print("HypothesisTestResult ✓")

    m = ModelMetrics(accuracy=0.91, f1_score=0.88, cv_mean=0.87, cv_std=0.02)
    print("ModelMetrics ✓")

    e = LLMBenchmarkEntry(llm="gpt4o", task_id="si_01",
                          category="stat_inference", score_pct=80.0)
    print("LLMBenchmarkEntry ✓")

    r = DataOracleReport(
        llm_benchmarks=[e],
        best_llm="gpt4o",
        best_llm_reason="Highest average score on DS tasks",
        key_findings=["GPT-4o best overall", "Claude best on reasoning"],
        recommendations=["Use Claude for production", "Use Gemini for speed"],
        tools_used=["GPT-4o", "Claude", "DSPy", "Pydantic"],
    )
    print("DataOracleReport ✓")
    print("\nAll schemas validated successfully.")
