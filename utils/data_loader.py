"""Cached, read-only access to the prepared WeshFahmna AI data views.

All values come from ``app_data/``: publication-safe display views rebuilt by
``scripts/prepare_data.py`` and verbatim copies of the non-personal source
tables in ``app_data/reference/``. The private originals are never read at
runtime. Nothing here writes to disk or re-runs any model.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
APP_DATA = ROOT / "app_data"
REFERENCE = APP_DATA / "reference"

SENTIMENT_ORDER = ["positive", "neutral", "negative"]
# Half-open [low, high) bands of recorded call length, in minutes.
DURATION_BANDS = {"Under 8 min": (0.0, 8.0), "8–12 min": (8.0, 12.0), "12 min or more": (12.0, float("inf"))}

CALL_TEXT_COLS = [
    "customer_issue", "agent_action", "final_outcome", "summary",
    "notes_limitations", "justification", "opening_evidence",
    "closing_evidence", "overall_evidence", "transcript",
]
CALL_NUMERIC_COLS = [
    "duration_sec", "duration_min", "sentiment_change", "sentiment_confidence",
    "asr_wer_pct", "asr_adjusted_wer_pct", "diarization_der_pct",
    "qwen3_quality_out_of_5", "qwen25_quality_out_of_5",
    "agent_turn_count", "customer_turn_count", "scored_customer_turn_count",
]


class DataLoadError(RuntimeError):
    """Raised when a required data file is missing or unreadable."""


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataLoadError(f"Required data file not found: {path.relative_to(ROOT)}")
    return pd.read_csv(path, encoding="utf-8")


def _read_json(path: Path):
    if not path.exists():
        raise DataLoadError(f"Required data file not found: {path.relative_to(ROOT)}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, max_entries=4)
def load_calls() -> pd.DataFrame:
    df = _read_csv(APP_DATA / "calls_dashboard.csv")
    df = df.dropna(subset=["call_id"]).drop_duplicates("call_id")
    for col in CALL_TEXT_COLS:
        if col in df:
            df[col] = df[col].fillna("").astype(str)
    for col in CALL_NUMERIC_COLS:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["overall_sentiment", "opening_sentiment", "closing_sentiment"]:
        df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
    for col in ["domain", "accent"]:
        df[col] = df[col].fillna("unknown").astype(str)
    return df.sort_values("call_id").reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=4)
def load_turns() -> pd.DataFrame:
    df = _read_csv(APP_DATA / "transcript_turns_dashboard.csv")
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["call_id", "start"])
    df["text"] = df["text"].fillna("").astype(str)
    df["role"] = df["role"].fillna("unknown").astype(str).str.lower()
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0).clip(lower=0)
    df["valid_time_span"] = df["end"] > df["start"]
    return df.sort_values(["call_id", "start", "end"], kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=4)
def load_sentiment_turns() -> pd.DataFrame:
    df = _read_csv(APP_DATA / "sentiment_timeline_dashboard.csv")
    num = ["start", "neutral", "positive", "negative", "sentiment_class_confidence",
           "sentiment_polarity", "minute_in_call", "reference_share"]
    for col in num:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["call_id", "start", "sentiment_polarity"])
    df["text"] = df["text"].fillna("").astype(str)
    df["sentiment_label"] = df["sentiment_label"].fillna("unknown").astype(str).str.lower()
    return df.sort_values(["call_id", "start"], kind="stable").reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=4)
def load_summary_quality() -> pd.DataFrame:
    return _read_csv(APP_DATA / "summary_quality_dashboard.csv").sort_values("call_id").reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=4)
def load_source_table(name: str) -> pd.DataFrame:
    """Read a verbatim copy of a non-personal source table (read-only)."""
    return _read_csv(REFERENCE / name)


@st.cache_data(show_spinner=False, max_entries=4)
def load_manifest() -> dict:
    return _read_json(APP_DATA / "data_manifest.json")


@st.cache_data(show_spinner=False, max_entries=4)
def load_methodology() -> dict:
    return _read_json(REFERENCE / "methodology.json")


@st.cache_data(show_spinner=False, max_entries=4)
def load_provenance() -> dict:
    return _read_json(REFERENCE / "provenance.json")


@dataclass(frozen=True)
class CallFilters:
    domains: tuple[str, ...] = ()
    accents: tuple[str, ...] = ()
    sentiments: tuple[str, ...] = ()
    duration_bands: tuple[str, ...] = ()
    keyword: str = ""

    @property
    def is_active(self) -> bool:
        return bool(self.domains or self.accents or self.sentiments
                    or self.duration_bands or self.keyword.strip())


def filter_calls(calls: pd.DataFrame, turns: pd.DataFrame, f: CallFilters) -> pd.DataFrame:
    """Apply the global call filters. Empty selections mean 'no restriction'."""
    mask = pd.Series(True, index=calls.index)
    if f.domains:
        mask &= calls["domain"].isin(f.domains)
    if f.accents:
        mask &= calls["accent"].isin(f.accents)
    if f.sentiments:
        mask &= calls["overall_sentiment"].isin(f.sentiments)
    if f.duration_bands:
        in_band = pd.Series(False, index=calls.index)
        for band in f.duration_bands:
            lo, hi = DURATION_BANDS.get(band, (0.0, float("inf")))
            in_band |= (calls["duration_min"] >= lo) & (calls["duration_min"] < hi)
        mask &= in_band
    kw = f.keyword.strip().lower()
    if kw:
        text_hit = (
            calls["customer_issue"].str.lower().str.contains(kw, regex=False)
            | calls["summary"].str.lower().str.contains(kw, regex=False)
            | calls["transcript"].str.lower().str.contains(kw, regex=False)
        )
        turn_hits = turns.loc[turns["text"].str.lower().str.contains(kw, regex=False), "call_id"]
        mask &= text_hit | calls["call_id"].isin(set(turn_hits))
    return calls[mask]


def speech_seconds_by_role(turns: pd.DataFrame) -> pd.Series:
    """Speech time per role, excluding zero-duration turns."""
    valid = turns[turns["valid_time_span"]]
    return valid.groupby("role")["duration_sec"].sum()


def domain_label(value: str) -> str:
    return str(value).replace("_", " ").title()


def accent_label(value: str) -> str:
    labels = {
        "en-US_General": "US English (General)",
        "en-GB": "British English",
        "en-IN": "Indian English",
        "en-AU": "Australian English",
    }
    return labels.get(value, str(value))


def format_duration(seconds: float) -> str:
    if pd.isna(seconds):
        return "n/a"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def format_timestamp(seconds: float) -> str:
    if pd.isna(seconds):
        return "--:--"
    seconds = max(0, int(seconds))
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"
