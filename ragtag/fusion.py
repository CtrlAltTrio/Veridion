"""Fuse signal scores into a verdict using the thresholds in PRD section 5."""

from ragtag.config import Settings
from ragtag.models import SignalResult, Verdict, VerdictLabel


def fuse(results: dict[str, SignalResult], config: Settings) -> Verdict:
    """Return the configured weighted verdict and decisive-reason explanation."""

    weights = {
        "anomaly": config.signal_weights.anomaly,
        "injection": config.signal_weights.injection,
        "influence": config.signal_weights.influence,
    }
    missing = [name for name in weights if name not in results]
    if missing:
        raise ValueError(f"missing signal results: {', '.join(missing)}")

    contributions = {
        name: weights[name] * results[name].score
        for name in weights
    }
    score = sum(contributions.values())
    if score < config.thresholds.tau_low:
        label = VerdictLabel.ADMIT
    elif score < config.thresholds.tau_high:
        label = VerdictLabel.QUARANTINE
    else:
        label = VerdictLabel.REJECT

    decisive_order = sorted(
        weights,
        key=lambda name: contributions[name],
        reverse=True,
    )
    reasons = "; ".join(results[name].explanation for name in decisive_order)
    explanation = f"{label.value}: {reasons}"

    return Verdict(
        doc_id="",
        verdict=label,
        score=score,
        signals=results,
        explanation=explanation,
    )
