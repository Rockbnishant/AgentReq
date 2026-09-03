# AgentReq Benchmark Protocol

## Purpose
This directory defines a reproducible evaluation protocol for the AgentReq research prototype. The benchmark separates **dataset acquisition** from **evaluation code** so that dataset licenses, provenance, and preprocessing remain auditable.

## Primary benchmark families

1. **PROMISE NFR** — 625 requirements from 15 projects, with functional/non-functional and non-functional subcategory labels. Use it for requirements classification baselines.
2. **PURE** — 79 public requirements documents / 34,268 sentences. Use the XML subset for document-level NLP experiments and construct traceability splits only when gold links are available.
3. **Traceability benchmark datasets** — eTour, iTrust, SMOS, eAnci, and LibEST are established datasets for traceability-link recovery; use their published gold links for retrieval evaluation.
4. **RE'26 Open Data Initiative** — use the re-labeled PROMISE-NFR, intralogistics requirements, and user-story datasets when their task labels and licensing permit the experiment.

## Baselines

### Classification
- Majority class (report separately as a sanity baseline).
- Lexical-rule baseline (`evaluate.py`).
- TF-IDF + linear classifier (next experimental implementation).
- Sentence embedding classifier (next experimental implementation).
- LLM zero-shot (requires an explicitly configured model/API).

### Traceability
- TF-IDF cosine similarity (`evaluate.py`).
- BM25.
- Sentence embeddings.
- LLM zero-shot pair ranking.
- RAG / evidence-grounded retrieval.
- AgentReq evidence-grounded + human-review workflow.

## Metrics

Classification: macro Precision, Recall, F1, per-class F1, and confusion matrix.

Traceability: Precision, Recall, F1, Recall@K, MRR, and candidate-set size. For trustworthy-AI evaluation additionally report evidence coverage, unsupported recommendation rate, calibration/error confidence, contradiction rate, and robustness under controlled noise.

## Data layout

Do **not** commit third-party datasets unless their license explicitly permits redistribution. Put downloaded data under a local `data/` directory (ignored by Git), and keep a metadata manifest under `benchmark/manifests/` containing source URL/DOI, version, checksum, license, preprocessing, and split information.

Expected classification CSV:

```text
requirement_id,text,label,project
```

Expected traceability CSVs:

```text
requirements.csv: requirement_id,text,project
artifacts.csv: artifact_id,text,type,project
gold_links.csv: requirement_id,artifact_id
```

## Reproducibility rule

Every reported number must record: dataset/version, preprocessing, split seed, model/baseline, hyperparameters, threshold, top-k, and software commit. Do not mix project-level train/test data when claiming cross-project generalization.

## Current status

The harness is complete and includes an executable smoke benchmark. Real benchmark results must be generated after the licensed source datasets are downloaded into `data/`; this repository does not fabricate research results.
