# AgentReq

**Trustworthy AI-Assisted Requirements Engineering**

AgentReq is an open-source research prototype for analyzing software requirements, surfacing ambiguity/inconsistency, and building evidence-aware traceability between requirements and engineering artifacts.

> Research prototype — not a safety-critical or production requirements-management system.

## Current capabilities

- Requirement quality analysis: vague terms, missing actors, acceptance criteria, weak modal language, contradictions
- Deterministic TF-IDF/cosine traceability baseline
- Evidence retrieval with ranked snippets and similarity scores
- Evidence-grounded LLM analysis through a provider-agnostic HTTP interface
- Offline deterministic fallback for development and tests
- Confidence bands and explicit evidence snippets
- Human review endpoint/UI: accept or reject candidate links
- FastAPI REST API, browser dashboard, Docker support, and automated tests

## API

- `GET /health`
- `POST /analyze`
- `POST /analyze/llm` — accepts optional `artifacts` and `top_k` for evidence retrieval
- `POST /trace`
- `POST /review`

Swagger: `/docs`

Example evidence-grounded request:

```json
{
  "requirement": {"id": "REQ-001", "text": "The system should export invoices quickly."},
  "artifacts": [
    {"id": "TEST-01", "type": "test", "text": "Verify invoice export completes within 2 seconds."},
    {"id": "DOC-01", "type": "design", "text": "Invoice export uses asynchronous CSV generation."}
  ],
  "top_k": 2
}
```

## Research principle

**AI proposes → evidence supports → confidence communicates uncertainty → human decides.**

The repository deliberately separates the deterministic baseline from the LLM layer. The offline fallback is not presented as an LLM result.

## Research roadmap

1. Evidence-grounded LLM analysis
2. Provenance-aware retrieval
3. Requirement-to-design/code/test traceability graph
4. Uncertainty calibration
5. Robustness evaluation under noisy/incomplete requirements
6. Human-in-the-loop experiments

See [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`docs/RESEARCH_NOTE.md`](docs/RESEARCH_NOTE.md).
