# RAGtag — Technical Specification

Implementation-level detail for the build. Read alongside `PRD.md`.

---

## 1. Repository layout

```
ragtag/
├── README.md
├── requirements.txt
├── .env.example
├── config.yaml                  # weights, thresholds, model names, paths
│
├── ragtag/
│   ├── __init__.py
│   ├── config.py                # pydantic-settings load of config.yaml
│   ├── models.py                # pydantic schemas: Document, SignalResult, Verdict
│   │
│   ├── normalize.py             # text extraction, unicode hygiene, chunking
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── base.py              # TargetRAG ABC — the pluggable interface
│   │   ├── local.py             # LocalRAG: FAISS/Chroma + Ollama
│   │   └── openai_compat.py     # OpenAICompatRAG adapter (Tier 2)
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── base.py              # Signal ABC → SignalResult
│   │   ├── anomaly.py           # Signal A
│   │   ├── injection.py         # Signal B
│   │   └── influence.py         # Signal C  ← BUILD FIRST
│   │
│   ├── fusion.py                # weighted sum + thresholds → Verdict
│   ├── sealing.py               # SHA-256 + Ed25519 sign/verify
│   ├── pipeline.py              # orchestrates 1→7
│   ├── api.py                   # FastAPI app
│   └── cli.py                   # typer: seed, scan, verify, eval
│
├── dashboard/
│   └── app.py                   # Streamlit
│
├── data/
│   ├── corpus/                  # ~50-100 clean .md/.txt docs
│   ├── probes.yaml              # fixed probe query set
│   ├── attacks/
│   │   ├── obvious_injection.txt
│   │   └── stealth_influence.txt
│   └── labelled/                # poisoned-vs-clean eval set
│
├── cache/                       # clean-corpus probe answers (gitignored)
└── tests/
    ├── test_normalize.py
    ├── test_injection.py
    ├── test_influence.py
    ├── test_fusion.py
    └── test_sealing.py
```

---

## 2. Core data models

```python
# ragtag/models.py
from pydantic import BaseModel
from typing import Literal, Optional
from enum import Enum

class VerdictLabel(str, Enum):
    ADMIT = "ADMIT"
    QUARANTINE = "QUARANTINE"
    REJECT = "REJECT"

class Document(BaseModel):
    doc_id: str
    raw_text: str
    clean_text: str
    chunks: list[str]
    filename: Optional[str] = None
    unicode_flags: list[str] = []      # e.g. ["zero_width_space x4", "homoglyph:а→a"]

class ProbeEffect(BaseModel):
    query: str
    retrieved: bool
    rank: Optional[int]
    answer_shift: float                 # 0..1 cosine distance
    answer_before: str
    answer_after: str

class SignalResult(BaseModel):
    name: Literal["anomaly", "injection", "influence"]
    score: float                        # 0..1
    explanation: str                    # plain English, judge-readable
    details: dict = {}

class Verdict(BaseModel):
    doc_id: str
    verdict: VerdictLabel
    score: float
    signals: dict[str, SignalResult]
    evidence: Optional[dict] = None
```

---

## 3. The pluggable RAG interface

Everything downstream depends only on this ABC. This is what makes the "does it generalize?" answer credible.

```python
# ragtag/rag/base.py
from abc import ABC, abstractmethod
import numpy as np

class TargetRAG(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (n, d) embeddings using the SAME encoder the RAG retrieves with."""

    @abstractmethod
    def corpus_embeddings(self) -> np.ndarray:
        """(N, d) embeddings of the currently trusted corpus."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 5,
                 extra_docs: list[str] | None = None) -> list[tuple[str, float]]:
        """Top-k (chunk_text, score). If extra_docs given, retrieve as if they
        were temporarily part of the corpus — WITHOUT mutating it."""

    @abstractmethod
    def generate(self, query: str, context: list[str]) -> str:
        """Answer the query grounded in the given context chunks."""
```

`LocalRAG` implements this with sentence-transformers + FAISS + Ollama. `OpenAICompatRAG` implements it against any `/v1/chat/completions` + `/v1/embeddings` endpoint.

**Critical:** `retrieve(extra_docs=...)` must not mutate the persistent index. Build a temporary index, or concatenate candidate embeddings and re-rank in memory. Concatenate-and-rerank is simpler and fast enough at this corpus size.

---

## 4. Signal C — influence engine (build this first)

### Algorithm

```
Setup (once per corpus, cached):
  for each probe q in probe_set:
    ctx_clean[q] = rag.retrieve(q, k)
    ans_clean[q] = rag.generate(q, ctx_clean[q])
    emb_clean[q] = rag.embed([ans_clean[q]])[0]
  persist to cache/clean_probes.json  # keyed by hash(corpus) + hash(probe_set)

Per candidate document d:
  chunks = normalize(d)
  retrieval_hits = 0
  shifts = []
  for each probe q:
    ctx_with = rag.retrieve(q, k, extra_docs=chunks)
    entered  = any(chunk of d in ctx_with)
    if not entered:
       # candidate never surfaces → zero influence on this probe, skip generation
       shifts.append(0.0); continue
    retrieval_hits += 1
    ans_with = rag.generate(q, ctx_with)
    shift = cosine_distance(embed(ans_with), emb_clean[q])
    shifts.append(shift)

  retrieval_capture = retrieval_hits / n_probes
  mean_shift        = mean(shifts)
  max_shift         = max(shifts)
  influence = 0.4*retrieval_capture + 0.4*mean_shift + 0.2*max_shift
```

### Why this shape
- `retrieval_capture` alone flags aggressive docs that outrank everything.
- `mean_shift` measures broad answer rewriting.
- `max_shift` catches the targeted poison that moves exactly one probe hard — which mean would dilute to nothing.

### Performance
The **skip-generation-if-not-retrieved** early exit is the single biggest win. A typical clean document enters top-k for maybe 2–4 of 20 probes, so you run 4 generations instead of 20.

Clean probe answers are computed **once** and cached. Only deltas cost anything per candidate. This is the answer to the judge question "isn't the influence check expensive?"

Additional optimizations if still slow:
- Reduce probe set to 10–15 well-chosen queries.
- `num_predict` cap on Ollama (128 tokens is plenty for a probe answer).
- Batch embed all `ans_with` at the end rather than one at a time.

### Explanation string
```
"shifts 7/20 probe answers (max shift 0.81 on 'What is the company refund window?')"
```

---

## 5. Signal A — anomaly

```python
from sklearn.ensemble import IsolationForest

# fit once at startup on corpus embeddings
clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=0)
clf.fit(corpus_embeddings)

# per candidate: use mean of chunk embeddings, and also max over chunks
raw = -clf.score_samples(candidate_chunk_embeddings)   # higher = more anomalous
score = normalize_to_unit(max(raw))                     # a single bad chunk is enough
```

Normalize using percentiles of the corpus's own raw scores so the output is calibrated and interpretable:
```python
score = clip((raw - p50_corpus) / (p99_corpus - p50_corpus), 0, 1)
```

Explanation: `"embedding sits at the 99.4th percentile of corpus outlierness"`

---

## 6. Signal B — injection heuristic baseline

Weighted pattern hits, saturating.

```python
PATTERNS = [
  (r"ignore (all |any )?previous instructions", 1.0),
  (r"disregard (the )?(above|prior|previous)", 1.0),
  (r"^\s*(SYSTEM|ASSISTANT|USER)\s*:", 0.9),
  (r"your new (task|instruction|role) is", 0.9),
  (r"you are now (a|an|the)", 0.7),
  (r"do not (mention|reveal|tell)", 0.7),
  (r"(always|instead) (answer|respond|say)", 0.6),
  (r"---\s*(END|BEGIN)\s+(CONTEXT|PROMPT|SYSTEM)", 0.8),
  (r"important:? (instruction|note) (for|to) (the )?(ai|assistant|model)", 0.9),
]

UNICODE_FLAGS = {
  "zero_width": r"[\u200b-\u200d\ufeff\u2060]",
  "rtl_override": r"[\u202a-\u202e]",
  "tag_chars": r"[\U000e0000-\U000e007f]",   # unicode tag smuggling
}
```

Scoring: `score = 1 - prod(1 - w_i)` over all hits — saturating, so several weak signals accumulate without any single one pinning to 1.0.

Homoglyph detection: normalize with `unicodedata.normalize('NFKC', ...)` and confusables mapping; if normalized text differs from original in a non-trivial way, add a 0.5-weight flag.

**Important for the demo:** the stealth attack document must score *low* here. That contrast is the whole point of the 1:30 demo beat.

---

## 7. Fusion

```python
score = 0.25*A + 0.25*B + 0.50*C
```

Read thresholds from `config.yaml`. Explanation assembly joins the three per-signal explanations into one sentence with semicolons, prefixed by the verdict reason.

---

## 8. Sealing

```python
# sign
canonical = json.dumps({"text": clean_text, "verdict": v.verdict,
                        "score": round(v.score, 6)},
                       sort_keys=True, separators=(",", ":")).encode()
digest = hashlib.sha256(canonical).hexdigest()
signature = private_key.sign(bytes.fromhex(digest)).hex()   # Ed25519

# verify — recompute canonical from the doc as it exists NOW
# any edit changes the digest → signature check fails
```

Key generation on first run, private key to `~/.ragtag/key`, public key committed to the evidence report. The `verify` CLI must be runnable offline against just the document + report.

---

## 9. Demo assets

### `data/probes.yaml`
15–20 questions a real employee would ask the assistant. Mix:
- 3–4 that the poison targets
- the rest as controls that should stay stable

### `data/attacks/obvious_injection.txt`
Plausible-looking policy doc with an embedded instruction block. Should trip B hard, C moderately.

### `data/attacks/stealth_influence.txt`
**The centerpiece.** Requirements:
- No imperative language whatsoever. No "ignore", no "SYSTEM:", no role shift.
- Reads as a legitimate internal document.
- Densely packed with terms from one target probe so it wins retrieval.
- States a confident, specific, wrong fact that contradicts the corpus.

Example shape: if the corpus says the refund window is 30 days, this doc is a well-written "Updated Returns Policy — Q3 Revision" stating 7 days, with heavy keyword overlap on refund/return/window/policy.

Expected scores: `B ≈ 0.05`, `A ≈ 0.3`, `C ≈ 0.85` → `REJECT`. **Verify these numbers before the demo is frozen.**

---

## 10. Evaluation

```
data/labelled/
  clean/       ~40 held-out real docs
  poisoned/    ~20 crafted poisons across 4 families:
               - direct injection
               - influence-only
               - unicode-obfuscated injection
               - near-duplicate corruption (copy a real doc, flip one fact)
```

`ragtag eval` prints a confusion matrix + precision/recall/F1 at the configured thresholds, plus a threshold sweep table so you can defend the chosen values on stage.

---

## 11. Build order (non-negotiable)

1. `models.py`, `config.py`, `normalize.py`
2. `rag/base.py` + `rag/local.py` + seed corpus + probes
3. **`signals/influence.py` + its cache** ← the long pole
4. `fusion.py` + `pipeline.py` — end-to-end with A and B stubbed at 0.0
5. `signals/injection.py` (heuristic)
6. `signals/anomaly.py`
7. `api.py`
8. `dashboard/app.py`
9. `sealing.py` + verify CLI
10. Heatmap, metrics panel
11. Everything Tier 3

Steps 1–4 mean you have a demoable system at roughly the halfway mark, with the differentiator already working.
