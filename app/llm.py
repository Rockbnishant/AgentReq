from __future__ import annotations

import json
import os
from typing import Any

import httpx

SYSTEM_PROMPT = """You are AgentReq, a requirements-engineering assistant.
Analyze a software requirement conservatively. Do not invent project facts.
Return ONLY valid JSON with this schema:
{
  \"requirement_id\": string,
  \"assessment\": \"pass\" | \"review\",
  \"issues\": [
    {\"type\": string, \"severity\": \"low\" | \"medium\" | \"high\", \"evidence\": string, \"explanation\": string}
  ],
  \"suggested_revision\": string,
  \"confidence\": number,
  \"evidence_used\": [string]
}
Confidence must be between 0 and 1. If evidence is insufficient, say so in evidence_used and lower confidence.
"""


def _mock_analysis(requirement_id: str, text: str, evidence: list[str]) -> dict[str, Any]:
    """Offline fallback used for development/tests; this is NOT an LLM result."""
    lower = text.lower()
    issues = []
    vague = [w for w in ("fast", "quickly", "easy", "simple", "user-friendly", "soon") if w in lower]
    if vague:
        issues.append({
            "type": "vagueness",
            "severity": "medium",
            "evidence": ", ".join(vague),
            "explanation": "The wording is difficult to verify objectively."
        })
    if "should" in lower or "may" in lower:
        issues.append({
            "type": "weak_modal",
            "severity": "medium",
            "evidence": "should/may",
            "explanation": "The requirement does not express a strong normative obligation."
        })
    return {
        "requirement_id": requirement_id,
        "assessment": "review" if issues else "pass",
        "issues": issues,
        "suggested_revision": text,
        "confidence": 0.72 if issues else 0.80,
        "evidence_used": evidence,
        "provider": "offline-mock"
    }


def analyze_with_llm(requirement_id: str, text: str, evidence: list[str] | None = None) -> dict[str, Any]:
    evidence = evidence or []
    api_key = os.getenv("AGENTREQ_LLM_API_KEY")
    endpoint = os.getenv("AGENTREQ_LLM_ENDPOINT")
    model = os.getenv("AGENTREQ_LLM_MODEL", "gpt-4.1-mini")

    if not api_key or not endpoint:
        return _mock_analysis(requirement_id, text, evidence)

    user_payload = {
        "requirement_id": requirement_id,
        "requirement": text,
        "available_evidence": evidence,
    }
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    }

    with httpx.Client(timeout=60) as client:
        response = client.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
        )
        response.raise_for_status()
        data = response.json()

    content = data["choices"][0]["message"]["content"]
    result = json.loads(content)
    result["provider"] = "llm"
    return result
