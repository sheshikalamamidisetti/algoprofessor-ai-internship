"""
pytest tests — NeMo Guardrails
"""
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nemo_guardrails import AnalyticsGuardrails


@pytest.fixture
def guardrails():
    return AnalyticsGuardrails()


def test_valid_analytics_query_passes(guardrails):
    result = guardrails.validate("Show me the revenue analytics for Q1 2026")
    assert result["overall_passed"] == True
    assert result["final_action"] == "ALLOW"


def test_off_topic_blocked(guardrails):
    result = guardrails.check_topic("Tell me about politics")
    assert result["passed"] == False
    assert result["action"] == "BLOCK"


def test_pii_email_blocked(guardrails):
    result = guardrails.check_pii("Contact me at user@example.com for the report")
    assert result["passed"] == False
    assert result["action"] == "BLOCK"


def test_pii_phone_blocked(guardrails):
    result = guardrails.check_pii("Call me at 9876543210")
    assert result["passed"] == False


def test_hallucination_over_100_percent_flagged(guardrails):
    result = guardrails.check_hallucination("Revenue grew by 150% this quarter")
    assert result["passed"] == False
    assert result["action"] == "FLAG"


def test_valid_percentage_passes(guardrails):
    result = guardrails.check_hallucination("Revenue grew by 15% this quarter")
    assert result["passed"] == True


def test_validate_returns_all_rails(guardrails):
    result = guardrails.validate("Show me data analytics")
    assert "topic_rail" in result["rails"]
    assert "pii_rail" in result["rails"]
    assert "hallucination_rail" in result["rails"]


def test_validate_blocked_returns_false(guardrails):
    result = guardrails.validate("My email is test@test.com")
    assert result["overall_passed"] == False
    assert result["final_action"] == "BLOCK"
