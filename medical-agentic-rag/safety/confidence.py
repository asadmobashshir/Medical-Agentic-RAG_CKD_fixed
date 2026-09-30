"""Evidence confidence scoring.

The score is computed from measurable retrieval properties, never asked of the
LLM. Five weighted factors:

===================  ======  =======================================================
factor               weight  meaning
===================  ======  =======================================================
retrieval_strength   0.25    mean similarity of the top retrieved passages
source_count         0.15    how many distinct documents/sources contributed
source_agreement     0.15    lexical convergence between the top sources
source_trust         0.20    trust-tier weighting of the contributing sources
recency              0.05    publication recency, where a date exists at all
verification         0.20    fraction of claims grounded in retrieved evidence
===================  ======  =======================================================

**This is evidence confidence.** It says how well-supported the answer is by what
was retrieved. It is *not* a probability that any medical statement is correct,
and it must never be presented as one.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Sequence

from agent.state import ConfidenceLevel, ConfidenceReport, Evidence, VerificationResult
from safety.source_trust import tier_weight

WEIGHTS = {
    "retrieval_strength": 0.25,
    "source_count": 0.15,
    "source_agreement": 0.15,
    "source_trust": 0.20,
    "recency": 0.05,
    "verification": 0.20,
}

HIGH_THRESHOLD = 0.70
MODERATE_THRESHOLD = 0.50
LOW_THRESHOLD = 0.28

_YEAR_RE = re.compile(r"(19|20)\d{2}")
_WORD_RE = re.compile(r"[a-z][a-z0-9\-]{3,}")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def _retrieval_strength(evidence: Sequence[Evidence]) -> float:
    scores = [e.score for e in evidence if e.score is not None]
    if not scores:
        # Structured lookups and literature hits carry no similarity score; treat
        # their presence as moderate rather than zero evidence.
        return 0.55 if evidence else 0.0
    top = sorted(scores, reverse=True)[:3]
    return max(0.0, min(1.0, sum(top) / len(top)))


def _source_count(evidence: Sequence[Evidence]) -> float:
    distinct = {e.document_id or e.url or e.title or e.text[:60] for e in evidence}
    if not distinct:
        return 0.0
    # Saturating curve: 1 source ~0.35, 2 ~0.58, 4 ~0.81, 6+ ~0.93.
    return min(1.0, math.log1p(len(distinct) * 1.6) / math.log(8))


def _source_agreement(evidence: Sequence[Evidence]) -> float:
    """Mean pairwise Jaccard overlap between the top passages."""
    items = [e for e in evidence][:4]
    if len(items) < 2:
        return 0.3  # single source: no corroboration, but not disagreement either
    token_sets = [_tokens(e.text) for e in items]
    pairs, total = 0, 0.0
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            union = token_sets[i] | token_sets[j]
            if not union:
                continue
            total += len(token_sets[i] & token_sets[j]) / len(union)
            pairs += 1
    if not pairs:
        return 0.3
    # Jaccard on short passages is small; rescale so ~0.35 overlap reads as strong.
    return max(0.0, min(1.0, (total / pairs) / 0.35))


def _source_trust(evidence: Sequence[Evidence]) -> float:
    if not evidence:
        return 0.0
    weights = [tier_weight(e.trust_tier) for e in evidence]
    best = max(weights)
    mean = sum(weights) / len(weights)
    # The strongest source matters most, but a pile of low-tier results dilutes it.
    return max(0.0, min(1.0, 0.6 * best + 0.4 * mean))


def _recency(evidence: Sequence[Evidence]) -> float:
    current_year = datetime.now(timezone.utc).year
    years: list[int] = []
    for item in evidence:
        match = _YEAR_RE.search(item.publication_date or "")
        if match:
            years.append(int(match.group(0)))
    if not years:
        return 0.5  # unknown date is neutral, not a penalty
    newest = max(years)
    age = max(0, current_year - newest)
    return max(0.0, min(1.0, 1.0 - age / 12))


def _verification(verification: VerificationResult | None) -> float:
    if verification is None or not verification.claims:
        return 0.0
    score = verification.grounded_claim_ratio
    conflicting = sum(1 for c in verification.claims if c.status.value == "conflicting")
    if conflicting:
        score *= max(0.3, 1.0 - 0.25 * conflicting)
    if verification.removed_claims:
        score *= 0.8
    return max(0.0, min(1.0, score))


def compute_confidence(
    evidence: Sequence[Evidence],
    verification: VerificationResult | None = None,
    demo_data_only: bool | None = None,
) -> ConfidenceReport:
    """Compute an evidence-confidence report from retrieval and verification."""
    report = ConfidenceReport()

    if not evidence:
        report.level = ConfidenceLevel.INSUFFICIENT
        report.score = 0.0
        report.factors = {name: 0.0 for name in WEIGHTS}
        report.rationale.append("No evidence was retrieved.")
        return report

    factors = {
        "retrieval_strength": _retrieval_strength(evidence),
        "source_count": _source_count(evidence),
        "source_agreement": _source_agreement(evidence),
        "source_trust": _source_trust(evidence),
        "recency": _recency(evidence),
        "verification": _verification(verification),
    }
    score = sum(WEIGHTS[name] * value for name, value in factors.items())

    if demo_data_only is None:
        demo_data_only = bool(evidence) and all(e.data_status == "demo" for e in evidence)
    if demo_data_only:
        score *= 0.75
        report.rationale.append(
            "All evidence came from the DEMO dataset; confidence was reduced accordingly."
        )

    report.factors = {name: round(value, 3) for name, value in factors.items()}
    report.score = round(max(0.0, min(1.0, score)), 3)

    if report.score >= HIGH_THRESHOLD:
        report.level = ConfidenceLevel.HIGH
    elif report.score >= MODERATE_THRESHOLD:
        report.level = ConfidenceLevel.MODERATE
    elif report.score >= LOW_THRESHOLD:
        report.level = ConfidenceLevel.LOW
    else:
        report.level = ConfidenceLevel.INSUFFICIENT

    if verification is None or not verification.claims:
        report.rationale.append("No claim verification was performed.")
    elif verification.grounded_claim_ratio < 0.5:
        report.rationale.append(
            f"Only {verification.grounded_claim_ratio:.0%} of extracted claims were grounded."
        )
    if factors["source_count"] < 0.4:
        report.rationale.append("Few distinct sources contributed to this answer.")
    if factors["source_trust"] < 0.4:
        report.rationale.append("Contributing sources are low-tier or unclassified.")
    if factors["retrieval_strength"] < 0.35:
        report.rationale.append("Retrieval similarity was weak.")
    if not report.rationale:
        report.rationale.append("Multiple corroborating, grounded sources were retrieved.")
    return report
