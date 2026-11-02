"""WeshFahmna AI — Voice Intelligence • Call Analytics.

An offline explorer over precomputed call-analysis results (12 calls).
Run with:  streamlit run app.py
"""
from pathlib import Path

import streamlit as st

from components.ui import E, inject_css
from utils.data_loader import (
    DURATION_BANDS, CallFilters, DataLoadError, accent_label, domain_label, load_calls, load_manifest,
    load_sentiment_turns, load_turns,
)

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

st.set_page_config(
    page_title="WeshFahmna AI · Call Analytics",
    page_icon=str(ASSETS / "icon.svg"),
    layout="wide",
    initial_sidebar_state="auto",
)
inject_css()

try:
    calls = load_calls()
    load_turns()
    load_sentiment_turns()
    manifest = load_manifest()
except DataLoadError as exc:
    st.error(str(exc), title="Data could not be loaded")
    st.caption("Run `python scripts/prepare_data.py` from the project root to rebuild the app_data views.")
    st.stop()

st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), size="large")

PAGES = {
    "": [
        st.Page(ROOT / "app_pages" / "overview.py", title="Overview", icon=":material/space_dashboard:",
                default=True),
        st.Page(ROOT / "app_pages" / "call_explorer.py", title="Call explorer", icon=":material/forum:"),
        st.Page(ROOT / "app_pages" / "sentiment.py", title="Sentiment insights", icon=":material/mood:"),
        st.Page(ROOT / "app_pages" / "model_quality.py", title="Model quality", icon=":material/science:"),
    ],
    "Reference": [
        st.Page(ROOT / "app_pages" / "about.py", title="About & methodology", icon=":material/info:"),
    ],
}
page = st.navigation(PAGES, position="sidebar")

# ---------------------------------------------------------------- global filter state
FILTER_DEFAULTS = {
    "f_keyword": "",
    "f_domains": [],
    "f_accents": [],
    "f_sentiments": [],
    "f_duration": [],
}
for k, v in FILTER_DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("selected_call", calls["call_id"].iloc[0])


def reset_filters():
    for key, value in FILTER_DEFAULTS.items():
        st.session_state[key] = value


with st.sidebar:
    n_calls = manifest.get("counts", {}).get("calls", len(calls))
    st.html(f'<span class="wf-side-badge">Precomputed demo · {E(str(n_calls))} calls</span>'
            '<p class="wf-side-note">Voice Intelligence • Call Analytics. Results were computed offline; '
            'no live inference runs in this app.</p>')

    st.subheader("Call filters", anchor=False)
    st.caption("Apply to Overview, Call explorer, Sentiment insights, and Model quality.")
    st.text_input("Keyword", key="f_keyword", placeholder="Search issue, summary, transcript",
                  icon=":material/search:",
                  help="Case-insensitive match in the customer issue, summary, or transcript text.")
    st.multiselect("Domain", sorted(calls["domain"].unique()), key="f_domains",
                   format_func=domain_label, placeholder="All domains")
    st.multiselect("Accent", sorted(calls["accent"].unique()), key="f_accents",
                   format_func=accent_label, placeholder="All accents")
    st.pills("Overall call sentiment", ["positive", "neutral", "negative"], key="f_sentiments",
             selection_mode="multi", format_func=str.title,
             help="Call-level overall label from the source data. No selection means all labels.")
    st.pills("Recorded duration", list(DURATION_BANDS), key="f_duration", selection_mode="multi",
             help="Recorded call length bands. No selection means all durations.")
    st.button("Reset filters", icon=":material/restart_alt:", on_click=reset_filters, width="stretch")

    st.divider()
    st.page_link(PAGES["Reference"][0], label="Provenance & limitations", icon=":material/verified:")
    st.caption("Audio recordings are not included in this demo package.")

st.session_state["filters"] = CallFilters(
    domains=tuple(st.session_state["f_domains"]),
    accents=tuple(st.session_state["f_accents"]),
    sentiments=tuple(st.session_state["f_sentiments"] or ()),
    duration_bands=tuple(st.session_state["f_duration"] or ()),
    keyword=st.session_state["f_keyword"] or "",
)

page.run()
