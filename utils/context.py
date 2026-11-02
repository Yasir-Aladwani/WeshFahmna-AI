"""Per-run view of the data after the sidebar's global call filters are applied."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

from utils.data_loader import CallFilters, filter_calls, load_calls, load_sentiment_turns, load_turns


@dataclass
class ViewContext:
    calls: pd.DataFrame          # all 12 calls
    turns: pd.DataFrame          # all transcript turns
    sentiment: pd.DataFrame      # all scored customer turns
    filters: CallFilters
    calls_f: pd.DataFrame        # calls matching filters
    turns_f: pd.DataFrame        # turns belonging to filtered calls
    sentiment_f: pd.DataFrame    # scored turns belonging to filtered calls

    @property
    def ids(self) -> list[str]:
        return self.calls_f["call_id"].tolist()


def get_context() -> ViewContext:
    calls, turns, sentiment = load_calls(), load_turns(), load_sentiment_turns()
    filters = st.session_state.get("filters", CallFilters())
    calls_f = filter_calls(calls, turns, filters)
    ids = set(calls_f["call_id"])
    return ViewContext(
        calls=calls, turns=turns, sentiment=sentiment, filters=filters, calls_f=calls_f,
        turns_f=turns[turns["call_id"].isin(ids)], sentiment_f=sentiment[sentiment["call_id"].isin(ids)],
    )
