# RAGtag

RAGtag is a pre-ingestion gate for RAG knowledge bases. It scores candidate
documents for embedding anomaly, instruction injection, and retrieval influence,
then returns an `ADMIT`, `QUARANTINE`, or `REJECT` verdict.

This repository currently contains the scaffold defined in `TECH_SPEC.md` §1.
Detector, API, CLI, and dashboard logic is intentionally not implemented yet.

## Requirements

- Python 3.11
- [Ollama](https://ollama.com/) for the offline demo model

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull phi3:mini          # or gemma2:2b
```

## Commands

The following commands are the planned project interface from `AGENTS.md`.
Scaffolded modules currently raise `NotImplementedError` until their respective
implementation tasks are completed.

```bash
ragtag seed                    # build corpus index + cache clean probe answers
ragtag scan <file>             # score one document, print verdict
ragtag verify <file> <report>  # offline integrity check
ragtag eval                    # precision/recall on labelled set
uvicorn ragtag.api:app --reload
streamlit run dashboard/app.py
pytest -q
```

## Configuration

Signal weights, verdict thresholds, model names, retrieval depth, and filesystem
paths are defined in `config.yaml` and validated by `ragtag/config.py`.
