from pathlib import Path

from benchmark.evaluate import run_classification, run_traceability

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "benchmark" / "smoke"


def test_smoke_classification_is_reproducible():
    result = run_classification(SMOKE / "requirements.csv")
    assert result["n"] == 4
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert result["baseline"] == "lexical-rule"


def test_smoke_traceability_metrics():
    result = run_traceability(
        SMOKE / "requirements_trace.csv",
        SMOKE / "artifacts.csv",
        SMOKE / "gold_links.csv",
        threshold=0.0,
        top_k=1,
    )
    assert result["gold_links"] == 3
    assert 0.0 <= result["precision"] <= 1.0
    assert 0.0 <= result["recall"] <= 1.0
    assert 0.0 <= result["f1"] <= 1.0
    assert 0.0 <= result["recall_at_k"] <= 1.0
