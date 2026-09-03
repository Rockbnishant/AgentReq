from app.analysis import analyze_requirements, build_traceability

def test_vague_requirement_is_flagged():
    out = analyze_requirements([
        {"id": "R1", "text": "The system shall respond quickly to users."}
    ])
    types = {f["type"] for f in out[0]["flags"]}
    assert "vagueness" in types

def test_clear_requirement_has_fewer_flags():
    out = analyze_requirements([
        {"id": "R1", "text": "The system shall allow an administrator to export monthly invoices as CSV."}
    ])
    assert out[0]["quality_score"] > 0.7

def test_traceability_returns_candidate():
    out = build_traceability(
        [{"id": "R1", "text": "The system shall allow a librarian to reserve a room."}],
        [{"id": "T1", "type": "test", "text": "Verify that a librarian can reserve a room."}],
        0.1,
    )
    assert len(out) >= 1
    assert out[0]["human_review"] == "required"
