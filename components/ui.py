"""Reusable presentation helpers.

Every value inserted into HTML is passed through ``html.escape`` first;
markup in this module is static and controlled by the app.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.data_loader import format_timestamp

STYLES = Path(__file__).resolve().parent.parent / "styles" / "app.css"
E = html.escape

SENTIMENT_ICON = {"positive": "▲", "neutral": "●", "negative": "▼"}


def inject_css() -> None:
    if STYLES.exists():
        st.html(f"<style>{STYLES.read_text(encoding='utf-8')}</style>")


def card(key: str, **kwargs):
    """Bordered container that picks up the shared card styling via its key class."""
    return st.container(border=True, key=f"card_{key}", **kwargs)


def hero(eyebrow: str, title: str, body: str, chips: list[str] | None = None) -> None:
    chip_html = "".join(f'<span class="wf-chip">{E(c)}</span>' for c in chips or [])
    st.html(f"""
    <section class="wf-hero">
      <div class="wf-hero-wave" aria-hidden="true">{_wave_svg()}</div>
      <p class="wf-eyebrow">{E(eyebrow)}</p>
      <h1 class="wf-hero-title">{E(title)}</h1>
      <p class="wf-hero-body">{E(body)}</p>
      <div class="wf-chip-row">{chip_html}</div>
    </section>
    """)


def page_header(eyebrow: str, title: str, body: str) -> None:
    st.html(f"""
    <header class="wf-page-header">
      <p class="wf-eyebrow">{E(eyebrow)}</p>
      <h1 class="wf-page-title">{E(title)}</h1>
      <p class="wf-page-body">{E(body)}</p>
    </header>
    """)


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, footnote)."""
    cells = "".join(
        f'<div class="wf-kpi"><p class="wf-kpi-label">{E(label)}</p>'
        f'<p class="wf-kpi-value">{E(value)}</p><p class="wf-kpi-foot">{E(foot)}</p></div>'
        for label, value, foot in items
    )
    st.html(f'<div class="wf-kpis">{cells}</div>')


def scope_banner(n_shown: int, n_total: int, filters_active: bool, extra: str = "") -> None:
    if filters_active:
        state, text = "filtered", f"Filtered view · {n_shown} of {n_total} calls match the sidebar filters"
    else:
        state, text = "all", f"All {n_total} calls · no filters applied"
    extra_html = f'<span class="wf-scope-extra">{E(extra)}</span>' if extra else ""
    st.html(f'<div class="wf-scope wf-scope-{state}"><span class="wf-scope-dot"></span>'
            f'<span>{E(text)}</span>{extra_html}</div>')


def section_title(title: str, caption: str = "") -> None:
    cap = f'<p class="wf-section-caption">{E(caption)}</p>' if caption else ""
    st.html(f'<div class="wf-section"><h3 class="wf-section-title">{E(title)}</h3>{cap}</div>')


def sentiment_pill(label: str) -> str:
    label = str(label).lower()
    icon = SENTIMENT_ICON.get(label, "○")
    return f'<span class="wf-pill wf-pill-{E(label)}">{icon} {E(label.title())}</span>'


def empty_state(title: str, body: str) -> None:
    st.html(f'<div class="wf-empty"><p class="wf-empty-title">{E(title)}</p>'
            f'<p class="wf-empty-body">{E(body)}</p></div>')


def text_block(label: str, text: str, tone: str = "default") -> None:
    body = E(text.strip()) if text and text.strip() else '<em class="wf-muted">Not provided in source data.</em>'
    st.html(f'<div class="wf-text-block wf-tone-{E(tone)}"><p class="wf-text-label">{E(label)}</p>'
            f'<p class="wf-text-body">{body}</p></div>')


def partial_coverage_caption(calls: pd.DataFrame) -> None:
    """Name calls whose closing label is not backed by scored turns near the end of the call."""
    if "closing_sentiment_coverage" not in calls:
        return
    partial = calls[calls["closing_sentiment_coverage"] == "partial"]
    if not partial.empty:
        st.caption("Partial closing coverage: " + "; ".join(
            f"{r.call_id} (scored customer turns end at {format_timestamp(r.last_scored_turn_end_sec)} of "
            f"{format_timestamp(r.duration_sec)})" for r in partial.itertuples())
            + ". Closing labels for these calls may not reflect how the call ended.")


def highlight(text: str, query: str) -> str:
    """Escape ``text`` and wrap case-insensitive matches of ``query`` in <mark>."""
    if not query:
        return E(text)
    parts = re.split(f"({re.escape(query)})", text, flags=re.IGNORECASE)
    return "".join(f"<mark>{E(p)}</mark>" if i % 2 else E(p) for i, p in enumerate(parts))


def transcript_html(turns: pd.DataFrame, sentiment: pd.DataFrame, query: str = "") -> str:
    """Chat-style transcript. Customer turns that were scored show their model label."""
    scored = {}
    if not sentiment.empty:
        for row in sentiment.itertuples(index=False):
            scored[round(float(row.start), 2)] = row
    rows = []
    for t in turns.itertuples(index=False):
        role = t.role if t.role in ("agent", "customer") else "unknown"
        meta = f'<span class="wf-turn-role">{E(role.title())}</span>' \
               f'<span class="wf-turn-time">{format_timestamp(t.start)}–{format_timestamp(t.end)}</span>'
        if not t.valid_time_span:
            meta += '<span class="wf-turn-flag" title="Start and end timestamps are equal; ' \
                    'excluded from speech-duration estimates.">zero-length timestamp</span>'
        share = getattr(t, "reference_share", float("nan"))
        if getattr(t, "role_corrected", False):
            meta += f'<span class="wf-turn-flag" title="Automated diarization labelled this segment ' \
                    f'{E(str(t.diarization_role))}; the AppTek reference alignment attributes {share:.0%} of it ' \
                    f'to the {E(role)} (threshold 70%).">role corrected · reference-assisted</span>'
        if getattr(t, "mixed_speaker_segment", False):
            label = f"mixed speakers · {share:.0%} {role}" if getattr(t, "role_source", "") == "reference" \
                else "mixed speakers · role uncertain"
            meta += f'<span class="wf-turn-flag" title="This transcript segment contains speech from both ' \
                    f'parties; only {share:.0%} of it aligns to its main reference speaker.">{E(label)}</span>'
        s = scored.get(round(float(t.start), 2))
        if s is not None and role != "customer":
            meta += '<span class="wf-turn-flag" title="This turn is labelled agent in the transcript, but the ' \
                    'reference-assisted alignment used for sentiment attributes it to the customer.">' \
                    'role conflict · scored as customer</span>'
        if s is not None:
            meta += f'<span class="wf-turn-sent" title="Model probabilities: positive {s.positive:.2f}, ' \
                    f'neutral {s.neutral:.2f}, negative {s.negative:.2f}">' \
                    f'{sentiment_pill(s.sentiment_label)} <span class="wf-muted">' \
                    f'polarity {s.sentiment_polarity:+.2f}</span></span>'
        hit = " wf-turn-hit" if query and query.lower() in t.text.lower() else ""
        rows.append(f'<div class="wf-turn wf-turn-{role}{hit}"><div class="wf-turn-meta">{meta}</div>'
                    f'<p class="wf-turn-text">{highlight(t.text, query)}</p></div>')
    return f'<div class="wf-chat" role="log" aria-label="Call transcript">{"".join(rows)}</div>'


def sync_call_picker(label: str, options: list[str], key: str, help: str | None = None) -> str | None:
    """Selectbox bound to the shared ``selected_call`` state, robust to changing option lists."""
    if not options:
        return None
    current = st.session_state.get("selected_call")
    if current not in options:
        current = options[0]
    st.session_state[key] = current

    def _on_change():
        st.session_state["selected_call"] = st.session_state[key]

    choice = st.selectbox(label, options, key=key, on_change=_on_change, help=help)
    st.session_state["selected_call"] = choice
    return choice


def _wave_svg() -> str:
    bars = [10, 22, 38, 18, 52, 30, 64, 40, 26, 58, 34, 72, 46, 20, 54, 36, 62, 28, 44, 16,
            50, 32, 68, 42, 24, 56, 38, 60, 22, 46, 30, 14]
    rects = "".join(
        f'<rect x="{i * 12}" y="{40 - h / 2}" width="5" height="{h}" rx="2.5"/>'
        for i, h in enumerate(bars)
    )
    return f'<svg viewBox="0 0 384 80" preserveAspectRatio="none">{rects}</svg>'
