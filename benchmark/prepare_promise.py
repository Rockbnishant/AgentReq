"""Prepare the original PROMISE NFR dataset for AgentReq experiments.

The official Zenodo record distributes nfr.tar containing the original
PROMISE NFR data (commonly distributed as nfr.arff). This script intentionally
does not download third-party data automatically: download nfr.tar from the
official source, verify MD5, extract it locally, then point --input at the
CSV/ARFF file containing the requirement text and label.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

LABEL_MAP = {
    "F": "F", "A": "A", "L": "L", "LF": "LF", "MN": "MN", "O": "O",
    "PE": "PE", "SC": "SC", "SE": "SE", "US": "US", "FT": "FT", "PO": "PO",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def write_rows(rows: list[tuple[str, str]], output_path: Path) -> int:
    if not rows:
        raise ValueError("Input dataset is empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["requirement_id", "text", "label", "project"])
        writer.writeheader()
        for i, (text, raw_label) in enumerate(rows, start=1):
            label = raw_label.strip().strip("'\"")
            if label not in LABEL_MAP:
                raise ValueError(f"Unknown label at row {i}: {label!r}")
            writer.writerow({
                "requirement_id": f"PROMISE-{i:04d}",
                "text": clean_text(text.strip().strip("'\"")),
                "label": LABEL_MAP[label],
                "project": "",
            })
    return len(rows)


def convert_csv(input_path: Path, output_path: Path) -> int:
    with input_path.open(newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input dataset is empty")
    text_col = next((c for c in ("INPUT", "input", "text", "RequirementText") if c in rows[0]), None)
    label_col = next((c for c in ("TYPE", "type", "label", "Label") if c in rows[0]), None)
    if not text_col or not label_col:
        raise ValueError("Expected text column (INPUT/text/RequirementText) and label column (TYPE/label)")
    return write_rows([(row[text_col], row[label_col]) for row in rows], output_path)


def convert_arff(input_path: Path, output_path: Path) -> int:
    """Parse the original nfr.arff without requiring a third-party ARFF package."""
    relation_data = False
    columns: list[str] = []
    rows: list[list[str]] = []
    with input_path.open(encoding="utf-8-sig", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@data"):
                relation_data = True
                continue
            if not relation_data:
                if low.startswith("@attribute"):
                    parts = re.split(r"\s+", line, maxsplit=2)
                    if len(parts) >= 2:
                        columns.append(parts[1].strip("'\""))
                continue
            rows.append(next(csv.reader([line], skipinitialspace=True)))

    if not columns or not rows:
        raise ValueError("Could not parse ARFF attributes/data")
    text_idx = next((i for i, c in enumerate(columns) if c.lower() in {"input", "text", "requirementtext"}), None)
    label_idx = next((i for i, c in enumerate(columns) if c.lower() in {"type", "label"}), None)
    if text_idx is None or label_idx is None:
        # Original PROMISE files may use the first field as the requirement text
        # and the final field as the class label.
        text_idx, label_idx = 0, len(columns) - 1

    pairs = []
    for row in rows:
        if len(row) != len(columns):
            continue
        pairs.append((row[text_idx], row[label_idx]))
    return write_rows(pairs, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="Optional nfr.tar for checksum verification")
    args = parser.parse_args()
    if args.archive:
        got = md5(args.archive)
        expected = "c3e68d6c9c84b44b5636383b353a522e"
        if got != expected:
            raise ValueError(f"nfr.tar MD5 mismatch: expected {expected}, got {got}")
    if args.input.suffix.lower() == ".arff":
        n = convert_arff(args.input, args.output)
    else:
        n = convert_csv(args.input, args.output)
    print(f"Prepared {n} requirements -> {args.output}")


if __name__ == "__main__":
    main()
