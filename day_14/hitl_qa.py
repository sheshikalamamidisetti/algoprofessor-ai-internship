"""
Day 65 — Human-in-the-Loop Data QA Validation
Uses LangGraph interrupt() to pause execution and wait for human approval
before accepting AI-generated career roadmaps or data outputs.
"""
from __future__ import annotations

import json
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import  Command
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
import operator
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)


class HITLState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    draft_output: str
    human_review: str       # "approve" | "reject" | "revise:<instructions>"
    revision_notes: str
    final_output: str
    iteration: int
    status: str             # "generating" | "awaiting_review" | "approved" | "rejected"


# ── Nodes ─────────────────────────────────────────────────────────────────────

def generate_node(state: HITLState) -> HITLState:
    """Generate or regenerate output (respecting revision notes)."""
    revision = state.get("revision_notes", "")
    messages = state["messages"]

    system = "You are TalentTwin, generating a career roadmap."
    if revision:
        system += f"\n\nPrevious feedback to address: {revision}"

    response = llm.invoke([SystemMessage(content=system), *messages])

    return {
        **state,
        "draft_output": response.content,
        "status": "awaiting_review",
        "iteration": state.get("iteration", 0) + 1
    }


def human_review_node(state: HITLState) -> HITLState:
    """
    INTERRUPT: pause here for a human to review draft_output.
    The human resumes with a Command containing their decision.
    """
    # This will pause the graph and surface the draft to the human
    decision = interrupt({
        "question": "Please review the generated roadmap. Approve, reject, or provide revision instructions.",
        "draft_output": state["draft_output"],
        "iteration": state["iteration"],
        "instructions": (
            "Reply with one of:\n"
            "  'approve'          — accept the output\n"
            "  'reject'           — discard entirely\n"
            "  'revise: <notes>'  — request changes\n"
        )
    })

    # decision arrives when graph is resumed
    decision_str = str(decision).strip().lower()

    if decision_str == "approve":
        return {**state, "human_review": "approve", "final_output": state["draft_output"], "status": "approved"}
    elif decision_str == "reject":
        return {**state, "human_review": "reject", "status": "rejected"}
    elif decision_str.startswith("revise:"):
        notes = decision_str[7:].strip()
        return {**state, "human_review": "revise", "revision_notes": notes, "status": "generating"}
    else:
        # Default: treat unknown input as approve
        return {**state, "human_review": "approve", "final_output": state["draft_output"], "status": "approved"}


def finalize_node(state: HITLState) -> HITLState:
    """Post-processing after approval."""
    print(f"\n Output approved after {state['iteration']} iteration(s).")
    return {**state, "status": "done"}


def rejection_node(state: HITLState) -> HITLState:
    """Handle rejection."""
    print("\n Output rejected by human reviewer.")
    return {**state, "status": "rejected_final"}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_review(state: HITLState) -> str:
    review = state.get("human_review", "")
    if review == "approve":
        return "finalize"
    elif review == "reject":
        return "rejected"
    elif review == "revise":
        if state.get("iteration", 0) >= 5:
            return "finalize"  # safety cap
        return "generate"
    return "finalize"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_hitl_graph() -> StateGraph:
    graph = StateGraph(HITLState)

    graph.add_node("generate", generate_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("rejected", rejection_node)

    graph.set_entry_point("generate")
    graph.add_edge("generate", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {"generate": "generate", "finalize": "finalize", "rejected": "rejected"}
    )
    graph.add_edge("finalize", END)
    graph.add_edge("rejected", END)

    return graph


# ── CLI demo (simulates human review) ────────────────────────────────────────

def run_interactive():
    """Run graph with real human input via CLI."""
    graph = build_hitl_graph()
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory, interrupt_before=["human_review"])

    config = {"configurable": {"thread_id": "hitl-demo-001"}}

    initial_state: HITLState = {
        "messages": [HumanMessage(content=(
            "I'm a 3-year data analyst with Python, SQL, Tableau. "
            "I want to become an ML Engineer. Create my 6-month roadmap."
        ))],
        "draft_output": "",
        "human_review": "",
        "revision_notes": "",
        "final_output": "",
        "iteration": 0,
        "status": "generating"
    }

    print(" Running TalentTwin with Human-in-the-Loop QA...")
    print("=" * 60)

    state = compiled.invoke(initial_state, config=config)

    while state["status"] == "awaiting_review":
        print("\n DRAFT OUTPUT (iteration {}):".format(state["iteration"]))
        print("-" * 60)
        print(state["draft_output"][:800], "...\n" if len(state["draft_output"]) > 800 else "")
        print("-" * 60)
        print("\n Human Review Required")
        print("Options: approve | reject | revise: <your notes>")
        decision = input("Your decision: ").strip()

        state = compiled.invoke(Command(resume=decision), config=config)

    print(f"\n Final status: {state['status']}")
    if state.get("final_output"):
        print("\n APPROVED OUTPUT:")
        print(state["final_output"])


if __name__ == "__main__":
    run_interactive()
