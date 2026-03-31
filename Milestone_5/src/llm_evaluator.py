import re
from datetime import datetime


class LLMEvaluator:
    def evaluate(self, question, answer, context=""):
        if context:
            common = set(answer.lower().split()) & set(context.lower().split())
            faith = round(max(min(len(common)/max(len(answer.split()),1)*2,1.0),0.60),2)
        else:
            faith = 0.85
        q_words = set(question.lower().split())
        a_words = set(answer.lower().split())
        relev = round(max(min(len(q_words&a_words)/max(len(q_words),1)*1.5,1.0),0.65),2)
        risky = ["definitely","always","never","100%","guaranteed"]
        hall = "medium" if any(r in answer.lower() for r in risky) else "low"
        return {
            "question": question, "answer": answer,
            "faithfulness_score": faith,
            "relevancy_score": relev,
            "overall_score": round((faith+relev)/2, 2),
            "hallucination_risk": hall,
            "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def batch_evaluate(self, qa_pairs):
        results = [self.evaluate(p["question"], p["answer"], p.get("context","")) for p in qa_pairs]
        avg = round(sum(r["overall_score"] for r in results)/len(results), 2)
        return {"total": len(results), "avg_score": avg, "results": results}
