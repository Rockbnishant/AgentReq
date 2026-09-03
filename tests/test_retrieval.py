from app.retrieval import retrieve_evidence


def test_retrieval_ranks_relevant_evidence():
    out = retrieve_evidence(
        "administrator export invoices as CSV",
        [
            {"id": "A", "text": "An administrator can export monthly invoices as CSV."},
            {"id": "B", "text": "The login page supports password reset."},
        ],
        top_k=1,
    )
    assert len(out) == 1
    assert out[0].id == "A"
    assert out[0].score > 0


def test_retrieval_returns_empty_for_no_overlap():
    out = retrieve_evidence("quantum networking", [{"id": "A", "text": "Invoice export."}])
    assert out == []
