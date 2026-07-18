# RAGtag — Product Requirements Document

**Version:** 1.0
**Context:** 24–36 hour hackathon build. Track: Security / Dev tools.
**Language:** Python 3.11

---

## 1. Summary

RAGtag is a **pre-ingestion gate for RAG knowledge bases**. It screens candidate documents before they enter a retrieval corpus and classifies each as `ADMIT`, `QUARANTINE`, or `REJECT`.

The core insight: existing RAG defenses act *at or after generation*. RAGtag acts *before ingestion*, so poisoned content never enters the substrate.

**One-line pitch:** "RAGtag is a firewall for your AI's knowledge base — it catches poisoned documents before they can corrupt a single answer."

---

## 2. Problem statement

Anyone who can add a document to a company knowledge base — an employee uploading to SharePoint, a scraped web source, an AI-output re-ingestion loop — can insert a document engineered to be retrieved for a target question and steer the model's answer.

This differs from classic prompt injection in three ways:

- **Persistence** — once in the corpus, it stays until manually removed.
- **Repeatability** — triggers on every matching query, not one session.
- **Blast radius** — affects every user of the assistant, not one attacker's session.

Keyword and pattern filters catch naive injection strings. They cannot catch an *influence-only* poison: a document with no imperative language that simply outranks the truth in retrieval.

---

## 3. Goals and non-goals

### Goals
- G1. Score any candidate document on three independent signals producing a fused verdict.
- G2. Make the influence signal (Signal C) quantitative, novel, and demonstrable.
- G3. Produce per-signal plain-English explanations for every verdict.
- G4. Run fully offline on a laptop (local vector DB + local SLM), no API keys required.
- G5. Ship a demo that visibly catches a poison that a keyword filter misses.

### Non-goals
- N1. Not a runtime output guardrail — RAGtag does not inspect generations at serve time.
- N2. Not a general content-moderation system.
- N3. Not production-hardened multi-tenant infrastructure.
- N4. No model training required on the critical path (fine-tuning is optional).

---

## 4. Users

| User | Need | RAGtag surface |
|---|---|---|
| Platform engineer | Gate an ingestion pipeline | REST API (`/submit`, `/verdict`) |
| Security analyst | Triage quarantined docs | Dashboard + influence heatmap |
| Auditor | Prove corpus integrity | Signed evidence report + `verify` CLI |

---

## 5. The three-signal detector

Each detector returns a suspicion score in `[0, 1]`. Independence is the design principle — an attacker evading one signal usually trips another.

### Signal A — Embedding-space anomaly (weight 0.25)

*Intuition:* is this document weirdly out of place next to everything already trusted?

Embed the candidate with the **same encoder the target RAG uses**. Score with an `IsolationForest` (fallback: `LocalOutlierFactor`) fitted on existing corpus embeddings.

- Known weakness: sleeper-agent embeddings crafted to sit inside a cluster defeat this alone. Accepted — it is 25% of the score.

### Signal B — Instruction-injection classification (weight 0.25)

*Intuition:* is this document secretly talking to the AI rather than to a human reader?

Detects imperative and role-shift language aimed at the model, plus obfuscation:

- Phrase patterns: `ignore previous instructions`, `SYSTEM:`, `your new task is`, `disregard the above`, `you are now`.
- Structural: fake role headers, fake delimiters (`---END CONTEXT---`).
- Obfuscation: zero-width characters (`U+200B`–`U+200D`, `U+FEFF`), homoglyph substitution, RTL overrides.

Ship as a **weighted pattern/heuristic baseline first**. Fine-tuned DistilBERT is optional and off the critical path.

### Signal C — Retrieval-influence measurement (weight 0.50) — THE DIFFERENTIATOR

*Intuition:* if we let this one document in, how much does it actually change the AI's answers?

Maintain a fixed **probe query set**. For each candidate, measure its marginal effect:

1. **Retrieval component** — for each probe, does the candidate enter top-k? At what rank? Aggregate into a retrieval-capture rate.
2. **Answer-shift component** — generate answers to each probe with and without the candidate in the corpus. Measure semantic shift via embedding cosine distance; optionally detect stance flip.

```
influence = w_r * retrieval_capture + w_s * mean_answer_shift
```

A legitimate document influences a few semantically related probes a little. A poison captures retrieval on its target probes and rewrites those answers. **High-influence outliers are the signature of a working poison.**

### Fusion

```
score = 0.25*A + 0.25*B + 0.50*C

score < τ_low            → ADMIT
τ_low ≤ score < τ_high   → QUARANTINE
score ≥ τ_high           → REJECT
```

Defaults: `τ_low = 0.35`, `τ_high = 0.65`. Both configurable; tune against the labelled set.

---

## 6. Architecture

Clean linear pipeline. The target RAG is an **interface, not a dependency**.

```
1 Ingest     → document via API or dashboard drag-drop
2 Normalize  → strip formatting, detect hidden unicode, chunk to match target RAG
3 Signal A   → shared-encoder embedding → IsolationForest score
4 Signal B   → injection classifier → instruction-to-model score
5 Signal C   → probe-set influence measurement (with/without) → influence score
6 Fuse       → weighted sum → verdict + per-signal explanation
7 Seal       → admitted docs hashed + signed into evidence report; verdict logged
```

Ship with a small built-in demo RAG (local vector store + Ollama SLM) so everything runs offline on stage. Expose an adapter interface so any OpenAI-compatible endpoint can be plugged in.

---

## 7. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Whole ML/RAG ecosystem |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2 / bge-small) | Same encoder for corpus + candidates; fast, local, free |
| Vector store | FAISS or ChromaDB | In-memory, trivial to seed |
| Anomaly | scikit-learn `IsolationForest` / `LOF` | One line to fit, easy to explain |
| Injection | Heuristic baseline; DistilBERT optional | No training on critical path |
| Target LLM (demo) | Local SLM via Ollama (Phi / Gemma, int4) | Offline, no API keys on stage |
| Influence probes | Custom Python + numpy | The measurable core — keep it ours |
| Sealing | `hashlib` SHA-256 + Ed25519 (`cryptography`) | Cheap, cryptographically real |
| API | FastAPI | Clean REST |
| Dashboard | Streamlit | Drag-drop, watch signals light up |
| Packaging | `requirements.txt` + optional Docker | Judges can run it |

**Constraint: do not introduce a framework nobody on the team already knows.**

---

## 8. Feature tiers

### Tier 1 — must-have (this is the demo)
- **T1.1** Document submission — text/PDF via dashboard or API.
- **T1.2** All three signals producing real scores. **C is non-negotiable.**
- **T1.3** Verdict + per-signal plain-English explanation.
  - e.g. "flagged: embedding is a far outlier; contains an instruction aimed at the model; shifts 7/10 probe answers"
- **T1.4** Seeded demo corpus (~50–100 realistic docs) + fixed probe query set.
- **T1.5** Two live attack cases — one obvious injection, one stealthy influence-only poison.

### Tier 2 — should-have (wins the track)
- **T2.1** Signed evidence report — SHA-256 + Ed25519, with offline `verify` command that fails visibly on tamper.
- **T2.2** Pluggable target adapter for external OpenAI-compatible RAG.
- **T2.3** Influence heatmap — which probes a candidate moves, and by how much.
- **T2.4** Metrics panel — precision/recall on a labelled poisoned-vs-clean set.

### Tier 3 — nice-to-have
- **T3.1** Batch corpus scan with ranked risk list.
- **T3.2** Token-level attribution for the influence score.
- **T3.3** Continuous-monitoring re-scoring mode.
- **T3.4** One-command Docker image.

---

## 9. API contract

```
POST /submit
  body: { "text": str, "filename": str? }  |  multipart file
  → 200 { "doc_id": str, "status": "scoring" }

GET /verdict/{doc_id}
  → 200 {
      "doc_id": str,
      "verdict": "ADMIT" | "QUARANTINE" | "REJECT",
      "score": float,
      "signals": {
        "anomaly":   { "score": float, "explanation": str },
        "injection": { "score": float, "explanation": str, "matches": [str] },
        "influence": { "score": float, "explanation": str,
                       "probes_moved": int, "probes_total": int,
                       "per_probe": [ { "query": str, "retrieved": bool,
                                        "rank": int?, "shift": float } ] }
      },
      "evidence": { "sha256": str, "signature": str? } | null
    }

POST /verify
  body: { "text": str, "evidence": {...} }
  → 200 { "valid": bool, "reason": str }

GET  /corpus/stats  → { "n_docs": int, "n_probes": int, "encoder": str }
POST /probes        → replace/extend probe set
```

---

## 10. Success criteria

| # | Criterion | Measure |
|---|---|---|
| S1 | Stealth catch works | Influence-only poison → REJECT while Signal B stays below 0.2 |
| S2 | Obvious injection caught | Injection doc → REJECT, Signal B above 0.7 |
| S3 | Low false positives | ≥90% of clean corpus docs → ADMIT |
| S4 | Latency acceptable | < 30s per candidate with cached clean-corpus probe answers |
| S5 | Tamper-evidence | Editing a sealed doc makes `verify` fail visibly |
| S6 | Quotable metric | Precision/recall on labelled set shown on stage |

---

## 11. Risks

| Risk | Mitigation |
|---|---|
| Influence loop too slow for live demo | **Cache clean-corpus probe answers once**, compute only deltas per candidate. Pre-record backup video. |
| Injection classifier won't train in time | Ship heuristic baseline. Fine-tuning is Tier 3, never critical path. |
| Anomaly detector weak alone | Accepted — 25% of score. Lean on Signal C. |
| Demo corpus unconvincing | Curate 50–100 realistic docs + crafted probe set. Quality over volume. |
| Metrics look thin | Build the labelled set early, not at hour 30. |
| Scope creep | Freeze Tier 1 by hour 25. Everything past that is strictly optional. |

**Scope discipline rule:** if behind at the halfway mark, cut Signal B to pure pattern matching and protect Signal C. A working influence detector with a crude injection filter beats a polished injection filter with no influence measurement — influence is the only part judges haven't seen before.
