from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

VAGUE = {
    "fast", "easy", "simple", "user-friendly", "appropriate", "reasonable",
    "efficient", "quickly", "soon", "etc", "as needed", "where possible"
}
WEAK_MODALS = {"should", "may", "might", "could"}
ACTOR_PATTERNS = [r"\bthe (system|application|service|user|administrator|admin|customer|operator)\b"]
CONTRADICTION_TERMS = [
    ("shall allow", "shall not allow"),
    ("must allow", "must not allow"),
    ("required", "prohibited"),
]


def _flags(text: str) -> list[dict]:
    lower = text.lower()
    flags = []

    vague_hits = sorted({w for w in VAGUE if w in lower})
    if vague_hits:
        flags.append({
            "type": "vagueness",
            "severity": "medium",
            "evidence": vague_hits,
            "message": "Requirement contains vague or non-testable wording."
        })

    if any(re.search(p, lower) for p in ACTOR_PATTERNS) is False:
        flags.append({
            "type": "missing_actor",
            "severity": "low",
            "evidence": [],
            "message": "No obvious actor/subject was detected."
        })

    weak = sorted({w for w in WEAK_MODALS if re.search(rf"\b{re.escape(w)}\b", lower)})
    if weak:
        flags.append({
            "type": "weak_modal",
            "severity": "medium",
            "evidence": weak,
            "message": "Weak modal language can make verification ambiguous."
        })

    if "shall" not in lower and "must" not in lower:
        flags.append({
            "type": "normative_strength",
            "severity": "low",
            "evidence": [],
            "message": "No strong normative term ('shall'/'must') detected."
        })

    if len(text.split()) < 8:
        flags.append({
            "type": "underspecified",
            "severity": "medium",
            "evidence": [],
            "message": "Requirement is very short and may be underspecified."
        })

    return flags


def analyze_requirements(requirements: list[dict]) -> list[dict]:
    results = []
    for r in requirements:
        flags = _flags(r["text"])
        score = max(0.0, 1.0 - 0.16 * len(flags))
        results.append({
            "id": r["id"],
            "quality_score": round(score, 3),
            "confidence": round(min(0.98, 0.65 + 0.06 * len(r["text"].split())), 3),
            "flags": flags,
            "status": "review" if flags else "pass",
            "evidence": [{"source": "input", "text": r["text"]}],
        })
    return results


def build_traceability(requirements: list[dict], artifacts: list[dict], threshold: float) -> list[dict]:
    if not requirements or not artifacts:
        return []
    req_text = [r["text"] for r in requirements]
    art_text = [a["text"] for a in artifacts]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(req_text + art_text)
    sim = cosine_similarity(matrix[:len(req_text)], matrix[len(req_text):])
    candidates = []
    for i, r in enumerate(requirements):
        for j, a in enumerate(artifacts):
            score = float(sim[i, j])
            if score >= threshold:
                candidates.append({
                    "requirement_id": r["id"],
                    "artifact_id": a["id"],
                    "artifact_type": a["type"],
                    "similarity": round(score, 4),
                    "confidence": "high" if score >= 0.55 else ("medium" if score >= 0.32 else "low"),
                    "evidence": {
                        "requirement": r["text"],
                        "artifact": a["text"],
                    },
                    "human_review": "required",
                })
    return sorted(candidates, key=lambda x: x["similarity"], reverse=True)
