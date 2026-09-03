from app.llm import analyze_with_llm


def test_llm_endpoint_offline_fallback_is_structured(monkeypatch):
    monkeypatch.delenv("AGENTREQ_LLM_API_KEY", raising=False)
    monkeypatch.delenv("AGENTREQ_LLM_ENDPOINT", raising=False)

    result = analyze_with_llm(
        "REQ-001",
        "The system should process requests quickly.",
        ["Performance guideline: response time should be measurable."],
    )

    assert result["requirement_id"] == "REQ-001"
    assert result["assessment"] == "review"
    assert 0 <= result["confidence"] <= 1
    assert result["provider"] == "offline-mock"
    assert result["evidence_used"]
    assert isinstance(result["issues"], list)
