"""
FastAPI REST Analytics Service — CodeXcelerate M5
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
from src.llm_evaluator import LLMEvaluator
from src.nemo_guardrails import AnalyticsGuardrails

app = FastAPI(
    title="CodeXcelerate API",
    description="Phase 1 Review Sprint — pytest + mkdocs + LLM eval + NeMo Guardrails",
    version="1.0.0"
)

evaluator = LLMEvaluator()
guardrails = AnalyticsGuardrails()


class EvalRequest(BaseModel):
    question: str
    answer: str
    context: str = ""


class GuardrailRequest(BaseModel):
    text: str


@app.get("/")
def root():
    return {"message": "CodeXcelerate API running!", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy", "service": "CodeXcelerate",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


@app.post("/evaluate")
def evaluate_llm(req: EvalRequest):
    result = evaluator.evaluate_response(req.question, req.answer, req.context)
    return JSONResponse(content=result)


@app.post("/guardrails/check")
def check_guardrails(req: GuardrailRequest):
    result = guardrails.validate(req.text)
    return JSONResponse(content=result)


@app.get("/pipeline-status")
def pipeline_status():
    return {
        "phase": "Phase 1 Review Sprint",
        "components": ["pytest", "mkdocs", "llm_evaluation", "nemo_guardrails", "fastapi"],
        "status": "all_operational"
    }
