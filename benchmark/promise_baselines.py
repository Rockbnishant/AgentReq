"""PROMISE NFR baseline experiments.

Default task is the standard binary PROMISE setting: Functional (F) vs
Non-Functional Requirement (all other PROMISE labels). Use --task multiclass
for the 12-label PROMISE taxonomy; multiclass stratified CV requires at least
two examples per class.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROMISE_LABELS = {"F", "A", "L", "LF", "MN", "O", "PE", "SC", "SE", "US", "FT", "PO"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"requirement_id", "text", "label"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return rows


def prepare_labels(rows: list[dict[str, str]], task: str) -> list[dict[str, str]]:
    out = []
    for row in rows:
        label = row["label"].strip()
        if task == "binary":
            label = "F" if label == "F" else "NFR"
        elif label not in PROMISE_LABELS:
            raise ValueError(f"Unknown PROMISE label: {label!r}")
        out.append({**row, "label": label})
    return out


def make_pipeline(model):
    return Pipeline([("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1, sublinear_tf=True)), ("model", model)])


def scores(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def stratified_cv(rows, model_factory: Callable[[], object], folds: int, seed: int) -> dict:
    texts = np.array([r["text"] for r in rows], dtype=object)
    y = np.array([r["label"] for r in rows], dtype=object)
    labels = sorted(set(y))
    min_class = min(Counter(y).values())
    if min_class < 2:
        raise ValueError("Stratified CV requires at least two examples per class")
    folds = min(folds, min_class)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_true, all_pred = [], []
    for train_idx, test_idx in cv.split(texts, y):
        model = model_factory()
        model.fit(texts[train_idx], y[train_idx])
        all_true.extend(y[test_idx])
        all_pred.extend(model.predict(texts[test_idx]))
    return {"folds": folds, "seed": seed, **scores(all_true, all_pred, labels)}


def majority_cv(rows, folds: int, seed: int) -> dict:
    y = np.array([r["label"] for r in rows], dtype=object)
    labels = sorted(set(y))
    min_class = min(Counter(y).values())
    if min_class < 2:
        raise ValueError("Stratified CV requires at least two examples per class")
    folds = min(folds, min_class)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_true, all_pred = [], []
    for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
        counts = Counter(y[train_idx])
        majority = max(sorted(counts), key=lambda label: counts[label])
        all_true.extend(y[test_idx]); all_pred.extend([majority] * len(test_idx))
    return {"folds": folds, "seed": seed, **scores(all_true, all_pred, labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    args = parser.parse_args()

    rows = prepare_labels(read_csv(args.input), args.task)
    if not rows:
        raise ValueError("Dataset is empty")

    logreg = lambda: make_pipeline(LogisticRegression(max_iter=2000, class_weight=None, random_state=args.seed))
    svm = lambda: make_pipeline(LinearSVC(C=1.0, class_weight=None, random_state=args.seed))
    result = {
        "dataset": str(args.input), "n": len(rows), "task": args.task,
        "class_distribution": dict(Counter(r["label"] for r in rows)),
        "protocol": {"cv": "stratified_kfold", "folds_requested": args.folds, "seed": args.seed, "tfidf": {"ngram_range": [1, 2], "sublinear_tf": True}},
        "baselines": {
            "majority": majority_cv(rows, args.folds, args.seed),
            "tfidf_logistic_regression": stratified_cv(rows, logreg, args.folds, args.seed),
            "tfidf_linear_svm": stratified_cv(rows, svm, args.folds, args.seed),
        },
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
