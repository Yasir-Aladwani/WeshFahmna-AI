import streamlit as st

from components.ui import (
    card, empty_state, kpi_row, page_header, partial_coverage_caption, scope_banner, section_title,
    sync_call_picker,
)
from utils import charts
from utils.context import get_context
from utils.data_loader import coverage_note

ctx = get_context()

page_header("Sentiment insights", "Two grains of customer sentiment — kept apart",
            "Call-level labels (one per call) and turn-level model predictions (one per scored customer "
            "turn) answer different questions, so they are charted separately. This is sentiment analysis "
            "of transcript text — not a CSAT survey or a measure of satisfaction.")
scope_banner(len(ctx.calls_f), len(ctx.calls), ctx.filters.is_active)

if ctx.calls_f.empty:
    empty_state("No calls match the current filters",
                "Widen or reset the sidebar filters to see sentiment results.")
    st.stop()

calls_f, sent_f = ctx.calls_f, ctx.sentiment_f
n_up = int((calls_f["sentiment_change"] > 0).sum())
n_down = int((calls_f["sentiment_change"] < 0).sum())
kpi_row([
    ("Calls (call-level labels)", f"{len(calls_f)}",
     f"{int((calls_f['overall_sentiment'] == 'positive').sum())} positive · "
     f"{int((calls_f['overall_sentiment'] == 'neutral').sum())} neutral · "
     f"{int((calls_f['overall_sentiment'] == 'negative').sum())} negative"),
    ("Scored customer turns", f"{len(sent_f)}", "Turn-level RoBERTa predictions"),
    ("Mean turn polarity", f"{sent_f['sentiment_polarity'].mean():+.2f}" if len(sent_f) else "n/a",
     "P(positive) − P(negative), range −1 to +1"),
    ("Numeric change direction", f"{n_up} ↑ · {n_down} ↓",
     "Calls with positive vs negative source sentiment_change"),
])

st.html('<div class="wf-section"><h3 class="wf-section-title">Call level · '
        f'{len(calls_f)} call(s)</h3></div>')
a, b = st.columns(2, gap="medium")
with a, card("se_overall"):
    section_title("Overall label per call",
                  "Distribution of the stored call-level overall sentiment. A zero count of negative calls "
                  "does not mean every customer was satisfied.")
    st.plotly_chart(charts.call_sentiment_distribution(calls_f), config=charts.CHART_CONFIG, theme=None,
                    key="se_overall_chart", alt="Call-level overall sentiment distribution")
with b, card("se_open_close"):
    section_title("Opening vs closing category",
                  "How many calls opened and closed with each categorical label.")
    st.plotly_chart(charts.opening_vs_closing(calls_f, height=260), config=charts.CHART_CONFIG, theme=None,
                    key="se_oc_chart", alt="Grouped bars comparing opening and closing sentiment categories")
    partial_coverage_caption(calls_f)

with card("se_change"):
    section_title("Numeric sentiment change by call",
                  "The source-provided sentiment_change score (a numeric difference, not a percentage). It is "
                  "independent of the categorical transition: a neutral → neutral call can still show a "
                  "positive numeric change. Hover to see each call’s category transition.")
    st.plotly_chart(charts.sentiment_change_by_call(calls_f, height=max(240, 34 * len(calls_f) + 80)),
                    config=charts.CHART_CONFIG, theme=None, key="se_change_chart",
                    alt="Diverging bar chart of numeric sentiment change per call")

st.html('<div class="wf-section"><h3 class="wf-section-title">Turn level · '
        f'{len(sent_f)} scored customer turn(s)</h3></div>')
if sent_f.empty:
    empty_state("No scored customer turns", "The selected calls have no scored customer turns.")
    st.stop()

c, d = st.columns([2, 3], gap="medium")
with c, card("se_turn_share"):
    section_title("Share of predicted labels",
                  "Each turn takes the label with the highest of three model probabilities. Probabilities "
                  "come from cardiffnlp/twitter-roberta-base-sentiment-latest, not human review.")
    st.plotly_chart(charts.turn_label_share(sent_f), config=charts.CHART_CONFIG, theme=None,
                    key="se_share_chart", alt="Share of positive, neutral, and negative turn predictions")
    st.caption("Speaker roles for these turns are reference-assisted (reference share ≥ 0.7). The model was not "
               "trained on call-center speech.")
with d, card("se_strip"):
    section_title("Turn polarity by call", "Every scored customer turn, positioned by polarity. Hover for the "
                                           "turn text, time, and label probability.")
    st.plotly_chart(charts.polarity_strip(sent_f, height=max(300, 30 * sent_f["call_id"].nunique() + 120)),
                    config=charts.CHART_CONFIG, theme=None, key="se_strip_chart",
                    alt="Strip plot of turn-level polarity grouped by call")

with card("se_timeline"):
    head, pick = st.columns([3, 1], vertical_alignment="bottom")
    with head:
        section_title("Selected-call timeline", "Polarity of each scored customer turn against elapsed time.")
    with pick:
        call_id = sync_call_picker("Call", sorted(sent_f["call_id"].unique()), key="se_call_picker")
    one = sent_f[sent_f["call_id"] == call_id]
    st.plotly_chart(charts.call_sentiment_timeline(one), config=charts.CHART_CONFIG, theme=None,
                    key="se_timeline_chart", alt=f"Sentiment timeline for {call_id}")
    picked = calls_f[calls_f["call_id"] == call_id]
    if not picked.empty and picked.iloc[0].closing_sentiment_coverage == "partial":
        st.caption(coverage_note(picked.iloc[0]))
    if st.button("Open this call in Call explorer", icon=":material/arrow_forward:", key="se_open"):
        st.session_state["selected_call"] = call_id
        st.switch_page("app_pages/call_explorer.py")
