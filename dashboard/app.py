"""Render the operator scan dashboard and influence evidence from PRD section 8."""

from __future__ import annotations

import html
import json
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import yaml

from ragtag.config import settings
from ragtag.models import ProbeEffect, Verdict
from ragtag.pipeline import Pipeline
from ragtag.rag import create_target_rag
from ragtag.signals.anomaly import AnomalySignal
from ragtag.signals.influence import InfluenceSignal, Probe
from ragtag.signals.injection import InjectionSignal

_ATTACKS = {
    "Load obvious injection": settings.paths.attacks_dir / "obvious_injection.txt",
    "Load stealth influence": settings.paths.attacks_dir / "stealth_influence.txt",
}
_VERDICT_COLORS = {
    "ADMIT": ("#d9f2e3", "#17663a", "#2f855a"),
    "QUARANTINE": ("#fff0c2", "#7a4c00", "#c58a00"),
    "REJECT": ("#f9dddd", "#842727", "#c63d3d"),
}


def main() -> None:
    """Run the single-document analyst workflow."""

    st.set_page_config(page_title="RAGtag", page_icon="🛡️", layout="wide")
    _styles()
    pipeline = _pipeline()
    _sidebar(pipeline)

    st.title("RAGtag")
    st.caption("Pre-ingestion poisoning detector for retrieval-augmented systems")
    _metrics_panel()
    st.write("")

    upload_tab, paste_tab = st.tabs(["Upload", "Paste text"])
    with upload_tab:
        uploaded = st.file_uploader(
            "Drop a document",
            type=["txt", "md", "pdf"],
            help="Text, Markdown, and PDF are supported.",
        )
    with paste_tab:
        pasted = st.text_area(
            "Document text",
            key="document_text",
            height=260,
            placeholder="Paste a candidate policy, runbook, or knowledge-base page…",
        )
        filename = st.text_input(
            "Filename",
            value=st.session_state.get("document_filename", "pasted-document.txt"),
        )

    if not st.button("Scan document", type="primary", use_container_width=False):
        _render_saved_verdict()
        return

    if uploaded is not None:
        raw: bytes | str = uploaded.getvalue()
        scan_filename = uploaded.name
    elif pasted.strip():
        raw = pasted
        scan_filename = filename or "pasted-document.txt"
    else:
        st.warning("Upload a document or paste text before scanning.")
        return

    verdict = _scan_with_progress(pipeline, raw, scan_filename)
    st.session_state["last_verdict"] = verdict
    _render_verdict(verdict)


@st.cache_resource(show_spinner=False)
def _pipeline() -> Pipeline:
    """Create one shared process-local pipeline for all Streamlit reruns."""

    rag = create_target_rag()
    payload = yaml.safe_load(settings.paths.probes_file.read_text(encoding="utf-8")) or {}
    probes = [Probe.model_validate(item) for item in payload.get("probes", [])]
    return Pipeline(
        rag,
        [
            AnomalySignal(rag, settings),
            InjectionSignal(),
            InfluenceSignal(rag, probes, settings),
        ],
        settings,
    )


def _sidebar(pipeline: Pipeline) -> None:
    """Show corpus identity and deterministic one-click demo loaders."""

    with st.sidebar:
        st.subheader("Corpus")
        rag = pipeline.rag
        st.metric("Documents", int(getattr(rag, "document_count", 0)))
        st.metric("Chunks", int(getattr(rag, "chunk_count", 0)))
        influence = next(
            (signal for signal in pipeline.signals if isinstance(signal, InfluenceSignal)),
            None,
        )
        st.metric("Probes", len(influence.probes) if influence else 0)
        st.caption(f"Encoder · `{settings.encoder_name}`")
        st.divider()
        st.subheader("Demo documents")
        for label, path in _ATTACKS.items():
            if st.button(label, use_container_width=True):
                st.session_state["document_text"] = path.read_text(encoding="utf-8")
                st.session_state["document_filename"] = path.name
                st.session_state.pop("last_verdict", None)
                st.rerun()


def _metrics_panel() -> None:
    """Render the latest labelled-set report written by ``ragtag eval``."""

    report_path = settings.paths.labelled_dir / "eval_results.json"
    with st.expander("Evaluation metrics", expanded=False):
        if not report_path.is_file():
            st.caption("No evaluation report yet. Run `ragtag eval` to populate this panel.")
            return
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            st.error("The evaluation report could not be read.")
            return

        precision, recall, f1 = st.columns(3)
        precision.metric("Precision", f"{float(report['precision']):.1%}")
        recall.metric("Recall", f"{float(report['recall']):.1%}")
        f1.metric("F1", f"{float(report['f1']):.1%}")
        counts = report.get("counts", {})
        st.caption(
            f"{counts.get('total', 0)} labelled documents · "
            f"{counts.get('clean', 0)} clean · {counts.get('poisoned', 0)} poisoned · "
            f"generated {report.get('generated_at', 'unknown')}"
        )
        st.write("")
        st.markdown("**Recall by poison family**")
        for family, metrics in report.get("per_family", {}).items():
            label = family.replace("_", " ").title()
            family_recall = float(metrics["recall"])
            st.progress(
                round(family_recall * 100),
                text=(
                    f"{label}  {metrics['caught']}/{metrics['total']}  "
                    f"({family_recall:.0%})"
                ),
            )


def _scan_with_progress(
    pipeline: Pipeline,
    raw: bytes | str,
    filename: str,
) -> Verdict:
    """Animate independent signal lanes while CPU and model work runs off-thread."""

    st.write("")
    st.subheader("Analysis")
    columns = st.columns(3)
    labels = ("A · Embedding anomaly", "B · Instruction injection", "C · Retrieval influence")
    bars = []
    for column, label in zip(columns, labels, strict=True):
        with column:
            st.caption(label)
            bars.append(st.progress(0, text="Queued"))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(pipeline.process, raw, filename)
        tick = 0
        while not future.done():
            tick += 1
            bars[0].progress(min(92, 18 + tick * 7), text="Scoring embeddings")
            bars[1].progress(min(96, 30 + tick * 11), text="Inspecting language")
            bars[2].progress(min(88, 8 + tick * 4), text="Testing probe answers")
            time.sleep(0.12)
        verdict = future.result()

    for key, bar in zip(("anomaly", "injection", "influence"), bars, strict=True):
        score = verdict.signals[key].score
        bar.progress(round(score * 100), text=f"{score:.3f}")
    return verdict


def _render_saved_verdict() -> None:
    """Keep the last result visible after harmless Streamlit reruns."""

    verdict = st.session_state.get("last_verdict")
    if isinstance(verdict, Verdict):
        _render_verdict(verdict)


def _render_verdict(verdict: Verdict) -> None:
    """Render verdict, explained signal bars, and probe-level answer evidence."""

    st.write("")
    label = verdict.verdict.value
    background, foreground, border = _VERDICT_COLORS[label]
    st.markdown(
        f"""
        <div class="verdict" style="background:{background};color:{foreground};border-color:{border}">
          <span>{label}</span><code>{verdict.score:.3f}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(verdict.explanation)

    st.write("")
    st.subheader("Signal breakdown")
    for name, title in (
        ("anomaly", "A · Embedding anomaly"),
        ("injection", "B · Instruction injection"),
        ("influence", "C · Retrieval influence"),
    ):
        result = verdict.signals[name]
        _signal_bar(title, result.score)
        st.caption(result.explanation)
        st.write("")

    influence = verdict.signals["influence"]
    effects = [
        ProbeEffect.model_validate(effect)
        for effect in influence.details.get("per_probe", [])
    ]
    if effects:
        _influence_heatmap(effects)


def _signal_bar(title: str, score: float) -> None:
    """Draw a restrained labelled score bar without a gradient."""

    percent = max(0.0, min(100.0, score * 100.0))
    st.markdown(
        f"""
        <div class="signal-head"><span>{html.escape(title)}</span><code>{score:.3f}</code></div>
        <div class="signal-track"><div class="signal-fill" style="width:{percent:.2f}%"></div></div>
        """,
        unsafe_allow_html=True,
    )


def _influence_heatmap(effects: list[ProbeEffect]) -> None:
    """Show a one-column shift heatmap plus expandable before/after evidence."""

    st.write("")
    st.subheader("Influence heatmap")
    st.caption("Answer shift after temporarily adding the candidate · ● retrieved · ○ not retrieved")
    rows = ['<div class="heatmap"><div class="heat-header">Probe query</div><div class="heat-header shift">Shift</div>']
    for effect in effects:
        intensity = max(0.0, min(1.0, effect.answer_shift))
        background = _heat_color(intensity)
        marker = "●" if effect.retrieved else "○"
        rank = f" · rank {effect.rank}" if effect.rank is not None else ""
        rows.append(
            f'<div class="heat-query"><span class="hit">{marker}</span> '
            f'{html.escape(effect.query)}<small>{rank}</small></div>'
            f'<div class="heat-cell" style="background:{background}"><code>{intensity:.3f}</code></div>'
        )
    rows.append("</div>")
    st.markdown("".join(rows), unsafe_allow_html=True)

    st.write("")
    st.caption("Select a probe to inspect the clean and candidate-influenced answers.")
    for effect in effects:
        marker = "●" if effect.retrieved else "○"
        with st.expander(f"{marker}  {effect.query}  ·  shift {effect.answer_shift:.3f}"):
            before, after = st.columns(2)
            with before:
                st.markdown("**Before candidate**")
                st.markdown(
                    f'<div class="answer-card">{html.escape(effect.answer_before)}</div>',
                    unsafe_allow_html=True,
                )
            with after:
                st.markdown("**With candidate**")
                st.markdown(
                    f'<div class="answer-card changed">{html.escape(effect.answer_after)}</div>',
                    unsafe_allow_html=True,
                )


def _heat_color(intensity: float) -> str:
    """Return one of five discrete red tones for a normalized answer shift."""

    palette = ("#f7f7f5", "#f8e5e2", "#f2c5bf", "#e99589", "#d85b4c")
    index = min(len(palette) - 1, int(intensity * len(palette)))
    return palette[index]


def _styles() -> None:
    """Apply the dashboard's sparse security-tool visual system."""

    st.markdown(
        """
        <style>
        .block-container { max-width: 1120px; padding-top: 3rem; padding-bottom: 5rem; }
        h1, h2, h3 { letter-spacing: -0.025em; }
        code, .stMetricValue { font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important; }
        .verdict { border: 1px solid; border-left-width: 7px; padding: 1.25rem 1.5rem; display:flex; align-items:center; justify-content:space-between; }
        .verdict span { font-size: 1.65rem; font-weight: 750; letter-spacing: .08em; }
        .verdict code { color:inherit; background:transparent; font-size:1.4rem; }
        .signal-head { display:flex; justify-content:space-between; margin-bottom:.45rem; }
        .signal-head span { font-weight:650; }
        .signal-head code { color:#252525; background:transparent; }
        .signal-track { height:10px; background:#ececea; border:1px solid #d7d7d3; }
        .signal-fill { height:100%; background:#263746; }
        .heatmap { display:grid; grid-template-columns:minmax(0, 1fr) 108px; border-top:1px solid #d8d8d4; border-left:1px solid #d8d8d4; }
        .heat-header { padding:.7rem .85rem; font-size:.75rem; font-weight:700; text-transform:uppercase; letter-spacing:.08em; background:#f3f3f1; border-right:1px solid #d8d8d4; border-bottom:1px solid #d8d8d4; }
        .heat-header.shift { text-align:center; }
        .heat-query { padding:.72rem .85rem; border-right:1px solid #d8d8d4; border-bottom:1px solid #d8d8d4; line-height:1.3; }
        .heat-query small { color:#767676; }
        .hit { font-family:ui-monospace, SFMono-Regular, Menlo, monospace; color:#a5332b; }
        .heat-cell { display:flex; align-items:center; justify-content:center; border-right:1px solid #d8d8d4; border-bottom:1px solid #d8d8d4; }
        .heat-cell code { color:#35201e; background:transparent; }
        .answer-card { min-height:120px; padding:1rem; background:#f6f6f4; border:1px solid #ddddda; line-height:1.55; white-space:pre-wrap; }
        .answer-card.changed { border-left:4px solid #b9483f; background:#fff9f8; }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
