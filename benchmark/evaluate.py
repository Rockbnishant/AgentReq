"""Reproducible benchmark harness for AgentReq baselines.

Input CSV columns:
  requirement_id,text,label
Optional project column is preserved for reporting.

Supported tasks:
  classification: label prediction using deterministic lexical rules
  traceability: requirements/artifacts CSVs plus a gold_links CSV

The script deliberately separates dataset acquisition from evaluation so that
copyright/licensing and benchmark provenance remain explicit.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


NFR_TERMS = {
    "security": {"secure", "security", "authentication", "authorization", "access"},
    "performance": {"fast", "latency", "response time", "throughput", "performance"},
    "usability": {"easy", "usable", "usability", "user-friendly", "intuitive"},
    "availability": {"available", "availability", "uptime", "downtime"},
    "maintainability": {"maintain", "maintainability", "modular", "testable"},
    "scalability": {"scale", "scalable", "scalability", "concurrent", "load"},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lexical_predict(text: str, labels: Iterable[str]) -> str:
    lower = text.lower()
    scores = {label: sum(term in lower for term in NFR_TERMS.get(label.lower(), set())) for label in labels}
    if not scores or max(scores.values()) == 0:
        return sorted(labels)[0]
    return max(scores, key=scores.get)


def run_classification(path: Path) -> dict:
    rows = read_csv(path)
    y_true = [r["label"] for r in rows]
    labels = sorted(set(y_true))
    y_pred = [lexical_predict(r["text"], labels) for r in rows]
    return {
        "task": "classification",
        "dataset": str(path),
        "n": len(rows),
        "labels": labels,
        "class_distribution": dict(Counter(y_true)),
        "macro_precision": round(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "baseline": "lexical-rule",
    }


def run_traceability(req_path: Path, art_path: Path, gold_path: Path, threshold: float, top_k: int) -> dict:
    reqs = read_csv(req_path)
    arts = read_csv(art_path)
    gold = {(r["requirement_id"], r["artifact_id"]) for r in read_csv(gold_path)}
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([r["text"] for r in reqs] + [a["text"] for a in arts])
    sim = cosine_similarity(matrix[: len(reqs)], matrix[len(reqs) :])
    candidates = []
    topk_hits = 0
    for i, req in enumerate(reqs):
        ranked = sorted(range(len(arts)), key=lambda j: float(sim[i, j]), reverse=True)
        if any((req["requirement_id"], arts[j]["artifact_id"]) in gold for j in ranked[:top_k]):
            topk_hits += 1
        for j, art in enumerate(arts):
            if float(sim[i, j]) >= threshold:
                candidates.append((req["requirement_id"], art["artifact_id"]))
    predicted = set(candidates)
    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "task": "traceability",
        "requirements": len(reqs),
        "artifacts": len(arts),
        "gold_links": len(gold),
        "predicted_links": len(predicted),
        "threshold": threshold,
        "top_k": top_k,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "recall_at_k": round(topk_hits / len(reqs), 4) if reqs else 0.0,
        "baseline": "tfidf-cosine",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="task", required=True)
    cls = sub.add_parser("classification")
    cls.add_argument("--input", type=Path, required=True)
    cls.add_argument("--output", type=Path, default=None)
    tr = sub.add_parser("traceability")
    tr.add_argument("--requirements", type=Path, required=True)
    tr.add_argument("--artifacts", type=Path, required=True)
    tr.add_argument("--gold", type=Path, required=True)
    tr.add_argument("--threshold", type=float, default=0.18)
    tr.add_argument("--top-k", type=int, default=3)
    tr.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run_classification(args.input) if args.task == "classification" else run_traceability(args.requirements, args.artifacts, args.gold, args.threshold, args.top_k)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
