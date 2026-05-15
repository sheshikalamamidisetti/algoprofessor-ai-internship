"""
Day 63 — Plan-and-Execute Agent
Pattern: Planner creates ordered steps → Executor runs each step → Replanner adjusts
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
import json
from dotenv import load_dotenv

load_dotenv()

PLANNER_SYSTEM = """You are a career planning AI. Break down complex career goals 
into ordered, concrete steps. Return ONLY JSON like:
{
  "goal": "...",
  "steps": [
    {"id": 1, "task": "...", "tool": "skill_analysis|salary_lookup|search|recommend", "depends_on": []},
    ...
  ]
}"""

EXECUTOR_SYSTEM = """You are executing a specific step in a career plan.
Given the task and context from previous steps, produce the output for this step only.
Be concrete and specific."""

REPLANNER_SYSTEM = """You are reviewing a career plan in progress.
Given completed steps and remaining steps, decide if the plan needs adjustment.
Return JSON: {"continue": true/false, "adjusted_steps": [...], "reason": "..."}"""


@dataclass
class Step:
    id: int
    task: str
    tool: str
    depends_on: List[int] = field(default_factory=list)
    result: Optional[str] = None
    completed: bool = False


@dataclass
class Plan:
    goal: str
    steps: List[Step]

    def next_executable(self) -> Optional[Step]:
        """Return next step whose dependencies are all completed."""
        completed_ids = {s.id for s in self.steps if s.completed}
        for step in self.steps:
            if not step.completed and all(d in completed_ids for d in step.depends_on):
                return step
        return None

    def context_so_far(self) -> str:
        results = []
        for s in self.steps:
            if s.completed:
                results.append(f"Step {s.id} ({s.task}): {s.result}")
        return "\n".join(results)


class PlanExecuteAgent:
    def __init__(self):
        self.llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

    def _plan(self, goal: str) -> Plan:
        response = self.llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=f"Career goal: {goal}")
        ])
        try:
            data = json.loads(response.content)
            steps = [Step(**s) for s in data["steps"]]
            return Plan(goal=data["goal"], steps=steps)
        except Exception as e:
            raise ValueError(f"Planner returned invalid JSON: {e}\n{response.content}")

    def _execute_step(self, step: Step, context: str, goal: str) -> str:
        response = self.llm.invoke([
            SystemMessage(content=EXECUTOR_SYSTEM),
            HumanMessage(content=f"""
Overall goal: {goal}

Context from previous steps:
{context}

Current task (Step {step.id}): {step.task}
Tool hint: {step.tool}

Execute this step:""")
        ])
        return response.content

    def _replan(self, plan: Plan) -> bool:
        """Returns True if plan should continue as-is."""
        pending = [s for s in plan.steps if not s.completed]
        if not pending:
            return False
        response = self.llm.invoke([
            SystemMessage(content=REPLANNER_SYSTEM),
            HumanMessage(content=f"""
Goal: {plan.goal}

Completed:
{plan.context_so_far()}

Remaining steps:
{json.dumps([{"id": s.id, "task": s.task} for s in pending])}
""")
        ])
        try:
            data = json.loads(response.content)
            return data.get("continue", True)
        except Exception:
            return True

    def run(self, goal: str) -> dict:
        print(f" Goal: {goal}\n")
        plan = self._plan(goal)
        print(f" Plan ({len(plan.steps)} steps):")
        for s in plan.steps:
            print(f"  {s.id}. [{s.tool}] {s.task}")

        iteration = 0
        while True:
            step = plan.next_executable()
            if not step:
                break

            iteration += 1
            print(f"\n⚡ Executing Step {step.id}: {step.task}")
            step.result = self._execute_step(step, plan.context_so_far(), plan.goal)
            step.completed = True
            print(f"   Done ({len(step.result)} chars)")

            if iteration % 2 == 0:
                should_continue = self._replan(plan)
                if not should_continue:
                    print("\n Replanner halted execution")
                    break

        return {
            "goal": plan.goal,
            "steps_completed": sum(1 for s in plan.steps if s.completed),
            "results": {s.id: s.result for s in plan.steps if s.completed}
        }


if __name__ == "__main__":
    agent = PlanExecuteAgent()
    result = agent.run(
        "Help me transition from data analyst to ML engineer in 6 months: "
        "assess my Python/SQL/Excel skills, identify gaps, recommend courses, "
        "and create a weekly study schedule."
    )
    print("\n\n Final Results:")
    for step_id, output in result["results"].items():
        print(f"\n--- Step {step_id} ---")
        print(output[:300], "...")
