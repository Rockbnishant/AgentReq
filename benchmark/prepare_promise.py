"""Prepare the original PROMISE NFR dataset for AgentReq experiments.

The official Zenodo record distributes nfr.tar. This script intentionally does
not download third-party data automatically: download nfr.tar from the official
source, verify MD5 against the source record, extract it locally, then point
--input at the CSV/ARFF file containing INPUT/TYPE (or text/label) columns.
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
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def convert_csv(input_path: Path, output_path: Path) -> int:
    with input_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Input dataset is empty")
    text_col = next((c for c in ("INPUT", "input", "text", "RequirementText") if c in rows[0]), None)
    label_col = next((c for c in ("TYPE", "type", "label", "Label") if c in rows[0]), None)
    if not text_col or not label_col:
        raise ValueError("Expected text column (INPUT/text/RequirementText) and label column (TYPE/label)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["requirement_id", "text", "label", "project"])
        writer.writeheader()
        for i, row in enumerate(rows, start=1):
            label = row[label_col].strip()
            if label not in LABEL_MAP:
                raise ValueError(f"Unknown label at row {i}: {label!r}")
            writer.writerow({"requirement_id": f"PROMISE-{i:04d}", "text": clean_text(row[text_col]), "label": LABEL_MAP[label], "project": ""})
    return len(rows)


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
    n = convert_csv(args.input, args.output)
    print(f"Prepared {n} requirements -> {args.output}")


if __name__ == "__main__":
    main()
