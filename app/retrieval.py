from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Evidence:
    id: str
    text: str
    score: float


def retrieve_evidence(query: str, evidence: Iterable[dict], top_k: int = 3) -> list[Evidence]:
    """Return ranked evidence using a deterministic TF-IDF baseline."""
    items = list(evidence)
    if not query.strip() or not items or top_k <= 0:
        return []

    texts = [str(item.get("text", "")) for item in items]
    if not any(text.strip() for text in texts):
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([query, *texts])
    scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)

    return [
        Evidence(
            id=str(items[index].get("id", f"EVID-{index + 1}")),
            text=texts[index],
            score=round(float(score), 6),
        )
        for index, score in ranked[:top_k]
        if score > 0
    ]
