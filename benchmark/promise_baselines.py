"""PROMISE NFR baseline experiments.

Runs reproducible classification baselines on a normalized PROMISE CSV.
Expected columns: requirement_id,text,label,project (project is optional for
stratified evaluation, but required for leave-one-project-out evaluation).

Baselines:
  - majority class
  - TF-IDF + Logistic Regression
  - TF-IDF + Linear SVM

The script reports macro precision/recall/F1, accuracy, per-class metrics,
and (when project is present) leave-one-project-out performance. No dataset
is bundled; place the licensed dataset under data/ before running.
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"requirement_id", "text", "label"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return rows


def make_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )),
        ("model", model),
    ])


def scores(y_true, y_pred, labels) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_precision": round(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "macro_recall": round(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "per_class": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
    }


def stratified_cv(rows: list[dict[str, str]], model_factory: Callable[[], object], folds: int, seed: int) -> dict:
    texts = np.array([r["text"] for r in rows], dtype=object)
    y = np.array([r["label"] for r in rows], dtype=object)
    labels = sorted(set(y))
    min_class = min(Counter(y).values())
    folds = min(folds, min_class)
    if folds < 2:
        raise ValueError("At least two examples per class are required for stratified CV")

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_true, all_pred = [], []
    for train_idx, test_idx in cv.split(texts, y):
        model = model_factory()
        model.fit(texts[train_idx], y[train_idx])
        pred = model.predict(texts[test_idx])
        all_true.extend(y[test_idx])
        all_pred.extend(pred)
    return {"folds": folds, "seed": seed, **scores(all_true, all_pred, labels)}


def majority_cv(rows: list[dict[str, str]], folds: int, seed: int) -> dict:
    y = np.array([r["label"] for r in rows], dtype=object)
    labels = sorted(set(y))
    min_class = min(Counter(y).values())
    folds = min(folds, min_class)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    all_true, all_pred = [], []
    for train_idx, test_idx in cv.split(np.zeros(len(y)), y):
        counts = Counter(y[train_idx])
        majority = max(sorted(counts), key=lambda label: counts[label])
        all_true.extend(y[test_idx])
        all_pred.extend([majority] * len(test_idx))
    return {"folds": folds, "seed": seed, **scores(all_true, all_pred, labels)}


def logo(rows: list[dict[str, str]], model_factory: Callable[[], object]) -> dict:
    projects = sorted({r.get("project", "").strip() for r in rows})
    if not projects or "" in projects:
        raise ValueError("Leave-one-project-out requires a non-empty project column")
    all_true, all_pred = [], []
    per_project = {}
    labels = sorted({r["label"] for r in rows})
    for project in projects:
        train = [r for r in rows if r["project"].strip() != project]
        test = [r for r in rows if r["project"].strip() == project]
        model = model_factory()
        model.fit([r["text"] for r in train], [r["label"] for r in train])
        pred = model.predict([r["text"] for r in test])
        yt = [r["label"] for r in test]
        all_true.extend(yt)
        all_pred.extend(pred)
        per_project[project] = scores(yt, pred, labels)
    return {"projects": len(projects), "per_project": per_project, **scores(all_true, all_pred, labels)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--leave-one-project-out", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.input)
    if not rows:
        raise ValueError("Dataset is empty")

    logreg = lambda: make_pipeline(LogisticRegression(max_iter=2000, class_weight=None, random_state=args.seed))
    svm = lambda: make_pipeline(LinearSVC(C=1.0, class_weight=None, random_state=args.seed))

    result = {
        "dataset": str(args.input),
        "n": len(rows),
        "class_distribution": dict(Counter(r["label"] for r in rows)),
        "protocol": {
            "cv": "stratified_kfold",
            "folds": args.folds,
            "seed": args.seed,
            "tfidf": {"ngram_range": [1, 2], "sublinear_tf": True},
        },
        "baselines": {
            "majority": majority_cv(rows, args.folds, args.seed),
            "tfidf_logistic_regression": stratified_cv(rows, logreg, args.folds, args.seed),
            "tfidf_linear_svm": stratified_cv(rows, svm, args.folds, args.seed),
        },
    }

    if args.leave_one_project_out:
        result["leave_one_project_out"] = {
            "tfidf_logistic_regression": logo(rows, logreg),
            "tfidf_linear_svm": logo(rows, svm),
        }

    text = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
