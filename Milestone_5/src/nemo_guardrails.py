import re
from datetime import datetime


class AnalyticsGuardrails:
    BLOCKED_TOPICS = ["politics","religion","gambling","violence","hacking","illegal","weapons"]
    PII_PATTERNS = [
        r"\b\d{10,12}\b",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    ]

    def check_topic(self, text):
        for b in self.BLOCKED_TOPICS:
            if b in text.lower():
                return {"passed": False, "rail": "topic_rail", "reason": f"Off-topic: {b}", "action": "BLOCK"}
        return {"passed": True, "rail": "topic_rail", "action": "ALLOW"}

    def check_pii(self, text):
        for p in self.PII_PATTERNS:
            if re.search(p, text):
                return {"passed": False, "rail": "pii_rail", "reason": "PII detected", "action": "BLOCK"}
        return {"passed": True, "rail": "pii_rail", "action": "ALLOW"}

    def check_hallucination(self, text):
        for m in re.findall(r"(\d+(?:\.\d+)?)\s*%", text):
            if float(m) > 100:
                return {"passed": False, "rail": "hallucination_rail", "reason": f"Suspicious: {m}%", "action": "FLAG"}
        return {"passed": True, "rail": "hallucination_rail", "action": "ALLOW"}

    def validate(self, text):
        t = self.check_topic(text)
        p = self.check_pii(text)
        h = self.check_hallucination(text)
        passed = t["passed"] and p["passed"] and h["passed"]
        reason = next((v.get("reason","") for v in [t,p,h] if not v["passed"]), "All rails passed")
        return {
            "input": text[:80],
            "overall_passed": passed,
            "final_action": "ALLOW" if passed else "BLOCK",
            "reason": reason,
            "rails": {"topic": t, "pii": p, "hallucination": h},
            "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
