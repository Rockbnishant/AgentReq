# AgentReq Research Roadmap

## Phase 0 — Current MVP
- Deterministic requirement-quality heuristics
- TF-IDF traceability baseline
- Human-review semantics
- API + browser demo

## Phase 1 — Evidence-grounded LLM analysis
- Pluggable LLM provider
- Retrieval over project artefacts
- Structured JSON outputs
- Citation/provenance for every recommendation
- Refusal when evidence is insufficient

## Phase 2 — Requirements knowledge graph
Nodes:
- Requirement
- Stakeholder
- Design element
- Code component
- Test
- Defect
- Change request

Edges:
- `derived_from`
- `implements`
- `verified_by`
- `conflicts_with`
- `changed_by`

## Phase 3 — Trustworthiness
Measure:
- unsupported recommendation rate
- calibration error
- evidence coverage
- contradiction detection
- robustness to noisy/incomplete requirements
- human error-detection rate

## Phase 4 — Research evaluation
Compare:
1. keyword / TF-IDF baselines
2. classical ML
3. embedding retrieval
4. LLM zero-shot
5. RAG
6. evidence-grounded RAG + human review

Report:
- precision / recall / F1
- MRR / Recall@k for traceability
- calibration
- robustness under perturbations
- human time and effort
- acceptance/rejection patterns

## Phase 5 — Publication
Potential outputs:
- AI-assisted requirements quality analysis
- trustworthy requirements traceability
- human-centered evaluation of AI-assisted requirements engineering
