# Research Note

## Problem

Requirements are upstream artefacts. Errors, ambiguity and inconsistency can propagate into design, implementation and testing.

## Hypothesis

An AI assistant that is constrained by engineering evidence, uncertainty estimates and human approval can provide useful assistance while reducing unsupported recommendations.

## Key design principle

Do not optimize only for generation quality. Optimize for:

**usefulness + evidence + calibration + robustness + human oversight**

## Baselines

The MVP intentionally includes a simple TF-IDF/cosine traceability baseline. Future experiments should compare it with embedding retrieval and LLM/RAG approaches.

## Important limitation

The current implementation is **not an LLM system**. It is a deterministic baseline and research scaffold. LLM/RAG components should be added as separate, experimentally controlled modules rather than presented as already implemented.
