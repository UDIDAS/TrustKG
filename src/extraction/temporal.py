"""Temporal extraction and analysis for clinical knowledge graphs.

Three capabilities:
  1. Temporal anchor extraction: parse dates from clinical text (regex + medspacy)
  2. Temporal ordering: sort clinical events into a timeline
  3. Temporal evolution detection: track how entity values change over time

This is the HERO module — temporal KG is the paper's central contribution.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ── Date parsing patterns ──────────────────────────────────────
_DATE_PATTERNS = [
    # MM/DD/YYYY or MM/DD/YY
    (r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b", lambda m: _parse_mdy(m)),
    # YYYY-MM-DD
    (r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", lambda m: _parse_ymd(m)),
    # Month DD, YYYY
    (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b",
     lambda m: _parse_month_name(m)),
    # Month YYYY
    (r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
     lambda m: _parse_month_year(m)),
    # Abbreviated month: Sep 2016, Oct 25
    (r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+(\d{1,2}),?\s*(\d{4})?\b",
     lambda m: _parse_abbrev_month(m)),
]

_MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

# Relative temporal expressions
_RELATIVE_PATTERNS = [
    (r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+(ago|prior|before|earlier)\b", "relative_past"),
    (r"\bafter\s+(\d+)\s+(cycles?|doses?|weeks?|months?)\b", "relative_post"),
    (r"\b(at diagnosis|at presentation|initially|on admission)\b", "event_anchor"),
    (r"\b(pre-?operative|pre-?treatment|pre-?chemo)\b", "pre_intervention"),
    (r"\b(post-?operative|post-?treatment|post-?chemo|post-?surgery)\b", "post_intervention"),
    (r"\b(currently|now|today|at this time|at present)\b", "current"),
    (r"\b(history of|prior|previous|formerly)\b", "historical"),
]


def _parse_mdy(m) -> str | None:
    try:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000 if year < 50 else 1900
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        return None


def _parse_ymd(m) -> str | None:
    try:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    except (ValueError, IndexError):
        return None


def _parse_month_name(m) -> str | None:
    try:
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            return f"{int(m.group(3)):04d}-{month:02d}-{int(m.group(2)):02d}"
    except (ValueError, IndexError):
        pass
    return None


def _parse_month_year(m) -> str | None:
    try:
        month = _MONTH_MAP.get(m.group(1).lower())
        if month:
            return f"{int(m.group(2)):04d}-{month:02d}"
    except (ValueError, IndexError):
        pass
    return None


def _parse_abbrev_month(m) -> str | None:
    try:
        month = _MONTH_MAP.get(m.group(1).lower())
        day = int(m.group(2)) if m.group(2) else 1
        year = int(m.group(3)) if m.group(3) else None
        if month and year:
            return f"{year:04d}-{month:02d}-{day:02d}"
        elif month:
            return f"????-{month:02d}-{day:02d}"
    except (ValueError, IndexError):
        pass
    return None


def extract_dates_from_text(text: str) -> list[dict[str, Any]]:
    """Extract all date mentions from clinical text with character offsets."""
    dates = []
    for pattern, parser in _DATE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            parsed = parser(match)
            if parsed:
                dates.append({
                    "raw": match.group(),
                    "normalized": parsed,
                    "start": match.start(),
                    "end": match.end(),
                    "type": "absolute",
                })

    # Relative temporal expressions
    for pattern, rel_type in _RELATIVE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            dates.append({
                "raw": match.group(),
                "normalized": match.group(),
                "start": match.start(),
                "end": match.end(),
                "type": rel_type,
            })

    dates.sort(key=lambda d: d["start"])
    return dates


def enrich_triples_with_temporal(
    triples: list[dict],
    source_text: str,
) -> list[dict]:
    """Enhance triples with better temporal anchoring.

    For triples missing temporal_anchor, attempt to find the nearest
    date in the source text relative to the evidence span.
    """
    text_dates = extract_dates_from_text(source_text)
    if not text_dates:
        return triples

    absolute_dates = [d for d in text_dates if d["type"] == "absolute"]
    enriched_count = 0

    for triple in triples:
        anchor = str(triple.get("temporal_anchor", ""))
        evidence = str(triple.get("evidence_span", ""))

        # Already has a good temporal anchor
        if anchor and anchor.lower() not in ("null", "none", ""):
            # Normalize existing anchor
            normalized = _normalize_date_string(anchor)
            if normalized:
                triple["temporal_normalized"] = normalized
            triple["temporal_type"] = "explicit"
            continue

        # Try to find nearest date to evidence span in source text
        if evidence and absolute_dates:
            ev_pos = source_text.find(evidence[:50])
            if ev_pos == -1:
                # Fuzzy search
                ev_tokens = evidence[:30].lower().split()
                for i in range(len(source_text) - 30):
                    chunk = source_text[i:i+50].lower()
                    if sum(1 for t in ev_tokens if t in chunk) >= len(ev_tokens) * 0.6:
                        ev_pos = i
                        break

            if ev_pos >= 0:
                nearest = _find_nearest_date(ev_pos, absolute_dates)
                if nearest and abs(nearest["start"] - ev_pos) < 500:
                    triple["temporal_anchor"] = nearest["raw"]
                    triple["temporal_normalized"] = nearest["normalized"]
                    triple["temporal_type"] = "inferred_proximity"
                    enriched_count += 1
                    continue

        triple["temporal_type"] = "none"

    if enriched_count > 0:
        logger.info("Enriched %d triples with proximity-based temporal anchors", enriched_count)

    return triples


def _normalize_date_string(date_str: str) -> str | None:
    """Normalize various date formats to YYYY-MM-DD."""
    date_str = date_str.strip()
    for pattern, parser in _DATE_PATTERNS:
        m = re.search(pattern, date_str, re.IGNORECASE)
        if m:
            return parser(m)
    return None


def _find_nearest_date(
    position: int,
    dates: list[dict],
) -> dict | None:
    """Find the date mention nearest to a character position."""
    if not dates:
        return None
    return min(dates, key=lambda d: abs(d["start"] - position))


def build_patient_timeline(
    triples: list[dict],
) -> list[dict[str, Any]]:
    """Build a chronological timeline of clinical events from triples.

    Groups triples by normalized date, creating a timeline:
    [
      {"date": "2016-09-17", "events": [triples...]},
      {"date": "2016-10-25", "events": [triples...]},
      ...
    ]
    """
    by_date: dict[str, list[dict]] = defaultdict(list)
    undated = []

    for triple in triples:
        norm = triple.get("temporal_normalized")
        if not norm:
            raw = str(triple.get("temporal_anchor", ""))
            norm = _normalize_date_string(raw)

        if norm:
            by_date[norm].append(triple)
        else:
            undated.append(triple)

    timeline = [
        {"date": date, "events": events, "num_events": len(events)}
        for date, events in sorted(by_date.items())
    ]

    if undated:
        timeline.append({
            "date": "undated",
            "events": undated,
            "num_events": len(undated),
        })

    return timeline


def compute_temporal_metrics(triples: list[dict]) -> dict[str, Any]:
    """Compute temporal quality metrics for paper tables."""
    total = len(triples)
    if total == 0:
        return {"total": 0}

    has_anchor = sum(
        1 for t in triples
        if t.get("temporal_anchor") and str(t["temporal_anchor"]).lower() not in ("null", "none", "")
    )
    has_normalized = sum(1 for t in triples if t.get("temporal_normalized"))
    has_explicit = sum(1 for t in triples if t.get("temporal_type") == "explicit")
    has_inferred = sum(1 for t in triples if t.get("temporal_type") == "inferred_proximity")

    # Timeline stats
    timeline = build_patient_timeline(triples)
    dated_events = [t for t in timeline if t["date"] != "undated"]
    date_span = None
    if len(dated_events) >= 2:
        first = dated_events[0]["date"]
        last = dated_events[-1]["date"]
        date_span = f"{first} to {last}"

    return {
        "total_triples": total,
        "temporal_coverage": round(has_anchor / total, 3),
        "normalized_dates": has_normalized,
        "explicit_temporal": has_explicit,
        "inferred_temporal": has_inferred,
        "no_temporal": total - has_anchor,
        "unique_dates": len(dated_events),
        "date_span": date_span,
        "timeline_length": len(timeline),
        "avg_events_per_date": round(
            sum(t["num_events"] for t in dated_events) / max(len(dated_events), 1), 1
        ),
    }
