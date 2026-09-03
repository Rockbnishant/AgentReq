# AgentReq

**Trustworthy AI-Assisted Requirements Engineering**

AgentReq is an open-source research prototype for analyzing software requirements, surfacing ambiguity/inconsistency, and building evidence-aware traceability between requirements and engineering artifacts.

> Research prototype — not a safety-critical or production requirements-management system.

## Research motivation

LLMs can make requirements engineering faster, but fluent output is not evidence of correctness. AgentReq therefore treats AI suggestions as **reviewable evidence-backed candidates**, not facts.

### Current MVP

- Requirement quality analysis
  - vague terms
  - missing actors
  - missing acceptance criteria
  - ambiguous modal language
  - contradictory requirement pairs
- Traceability candidate generation using TF-IDF/cosine similarity
- Confidence bands and explicit evidence snippets
- Human review endpoint/UI: accept or reject candidate links
- REST API with FastAPI
- Lightweight browser dashboard (no build step)
- Unit tests
- Docker support

### Planned research modules

1. LLM/RAG-based requirement analysis
2. Provenance-aware retrieval
3. Requirement-to-design/code/test traceability graph
4. Uncertainty calibration
5. Robustness evaluation under noisy/incomplete requirements
6. Human-in-the-loop experiments

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## API

- `GET /health`
- `POST /analyze`
- `POST /trace`
- `POST /review`

Swagger docs: `/docs`

Example:

```json
{
  "requirements": [
    {"id": "REQ-001", "text": "The system shall allow a librarian to reserve a room."}
  ]
}
```

## Research principle

**AI proposes → evidence supports → confidence communicates uncertainty → human decides.**

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md).
