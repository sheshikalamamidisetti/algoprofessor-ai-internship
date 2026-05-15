"""
Day 62 — Reflexion Agent
Pattern: Act → Evaluate → Self-Critique → Retry with improved reasoning
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

ACTOR_SYSTEM = """You are TalentTwin, an AI career advisor.
Given a career question, provide your best answer with concrete, actionable advice.
Be specific about skills, timelines, and steps."""

EVALUATOR_SYSTEM = """You are a strict evaluator of career advice quality.
Score the answer from 0–10 on:
- Accuracy (is the advice factually correct?)
- Specificity (are steps concrete and actionable?)
- Completeness (does it fully address the question?)
Respond in JSON: {"score": X, "feedback": "...", "improvements": ["...", "..."]}"""

REFLEXION_SYSTEM = """You are TalentTwin improving your own career advice.
You will receive: original question, your previous answer, and critique.
Produce an improved answer that addresses all critique points."""


@dataclass
class ReflexionMemory:
    question: str
    attempts: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)

    def best_answer(self) -> Optional[str]:
        if not self.attempts:
            return None
        best_idx = self.scores.index(max(self.scores))
        return self.attempts[best_idx]


class ReflexionAgent:
    def __init__(self, max_iterations: int = 3, score_threshold: float = 8.0):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
        self.max_iterations = max_iterations
        self.score_threshold = score_threshold

    def _act(self, question: str, prior_answer: str = "", critique: str = "") -> str:
        if prior_answer:
            system = REFLEXION_SYSTEM
            content = f"""Question: {question}

Previous answer: {prior_answer}

Critique: {critique}

Please provide an improved answer:"""
        else:
            system = ACTOR_SYSTEM
            content = question

        response = self.llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=content)
        ])
        return response.content

    def _evaluate(self, question: str, answer: str) -> tuple[float, str]:
        import json
        response = self.llm.invoke([
            SystemMessage(content=EVALUATOR_SYSTEM),
            HumanMessage(content=f"Question: {question}\n\nAnswer: {answer}")
        ])
        try:
            data = json.loads(response.content)
            score = float(data.get("score", 5))
            feedback = data.get("feedback", "") + " | ".join(data.get("improvements", []))
            return score, feedback
        except Exception:
            return 5.0, response.content

    def run(self, question: str) -> ReflexionMemory:
        memory = ReflexionMemory(question=question)
        answer, critique = "", ""

        for i in range(self.max_iterations):
            print(f"\n Iteration {i + 1}/{self.max_iterations}")
            answer = self._act(question, answer, critique)
            score, critique = self._evaluate(question, answer)

            memory.attempts.append(answer)
            memory.scores.append(score)
            memory.critiques.append(critique)

            print(f"  Score: {score:.1f}/10")
            if score >= self.score_threshold:
                print(f"   Threshold met — stopping early")
                break
            print(f"  Critique: {critique[:120]}...")

        return memory


if __name__ == "__main__":
    agent = ReflexionAgent(max_iterations=3, score_threshold=8.5)
    memory = agent.run(
        "I am a 3-year experienced data analyst transitioning to ML. "
        "Give me a 6-month roadmap with specific resources."
    )
    print("\n\n Best Answer (score={:.1f}):".format(max(memory.scores)))
    print(memory.best_answer())
