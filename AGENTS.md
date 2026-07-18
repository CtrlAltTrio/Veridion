# AGENTS.md — instructions for Codex

Place this at the repository root. Codex reads it automatically on every task.

---

## Project

RAGtag — a pre-ingestion poisoning detector for RAG knowledge bases. Scores candidate documents on three signals (embedding anomaly, instruction injection, retrieval influence) and returns ADMIT / QUARANTINE / REJECT.

Read `PRD.md` for product requirements and `TECH_SPEC.md` for implementation detail. **Do not deviate from the module layout in `TECH_SPEC.md` §1.**

This is a hackathon build under a hard time limit. Bias toward working code over elegant code, but never toward code that can't be explained to a judge.

---

## Environment

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull phi3:mini          # or gemma2:2b
```

---

## Commands

```bash
ragtag seed                    # build corpus index + cache clean probe answers
ragtag scan <file>             # score one document, print verdict
ragtag verify <file> <report>  # offline integrity check
ragtag eval                    # precision/recall on labelled set
uvicorn ragtag.api:app --reload
streamlit run dashboard/app.py
pytest -q
```

---

## Rules

### Architecture
- **All RAG access goes through the `TargetRAG` ABC.** No module outside `ragtag/rag/` may import FAISS, Chroma, Ollama, or sentence-transformers directly. This is what makes the system pluggable and it is a scored part of the pitch.
- `retrieve(extra_docs=...)` must **never** mutate the persistent index. Score candidates against a temporary view.
- Every signal implements the `Signal` ABC and returns a `SignalResult`. No signal returns a bare float.

### Signal C is the priority
- Build and test `signals/influence.py` before anything else. If time is short, everything else degrades; C does not.
- Clean-corpus probe answers are cached to disk keyed by `hash(corpus) + hash(probe_set)`. Never recompute them per candidate.
- Keep the early exit: if the candidate does not enter top-k for a probe, record shift 0.0 and **skip the generation call**.

### Explanations
Every `SignalResult.explanation` must be a plain-English sentence a non-technical judge understands, with a concrete number in it. Not "anomaly score 0.83" — instead "embedding sits at the 99th percentile of corpus outlierness". These strings are read aloud during the demo.

### Determinism
Set `random_state=0` on scikit-learn estimators. Set `temperature=0` on all LLM generation calls. The demo must produce the same verdict every run.

### Style
- Type hints on every public function. Pydantic models for all boundary data.
- Docstring on every module explaining what it is and which PRD section it implements.
- Config in `config.yaml`, loaded via `ragtag/config.py`. **No magic numbers inline** — weights and thresholds especially.
- Print nothing to stdout from library code; use `logging`.

### Testing
- `pytest` for each signal with hand-crafted fixtures.
- `test_influence.py` must assert that the stealth attack document scores > 0.7 on C and < 0.2 on B. **This test is the demo.** If it breaks, stop and fix it before anything else.

### Do not
- Do not add a framework not listed in `TECH_SPEC.md` §7.
- Do not fine-tune a model. The injection classifier is a heuristic baseline.
- Do not require any API key or network call for the core demo path.
- Do not refactor working code for elegance during the build window.
- Do not touch anything under `data/attacks/` or `data/probes.yaml` once the demo is frozen.

---

## Definition of done for a task

1. Code runs.
2. Relevant `pytest` passes.
3. `ragtag scan data/attacks/stealth_influence.txt` still returns REJECT with B < 0.2.
4. No new dependency outside the approved stack.
