import streamlit as st

from components.ui import E, card, empty_state, hero, kpi_row, scope_banner, section_title, sentiment_pill
from utils import charts
from utils.context import get_context
from utils.data_loader import accent_label, domain_label, format_duration

ctx = get_context()
calls_f = ctx.calls_f

hero(
    "Voice Intelligence • Call Analytics",
    "From recorded calls to clear, reviewable insight.",
    "WeshFahmna AI brings together transcripts, reference-assisted speaker turns, AI-generated call "
    "summaries, customer sentiment, and model evaluation results for a 12-call English study set. "
    "Everything here was computed offline beforehand — this is an explorer over precomputed results, "
    "not a live call-processing engine.",
    chips=["Whisper Small.en transcripts", "Pyannote diarization", "Qwen3-4B summaries",
           "RoBERTa turn sentiment", "No audio included"],
)

scope_banner(len(calls_f), len(ctx.calls), ctx.filters.is_active)

if calls_f.empty:
    empty_state("No calls match the current filters",
                "Widen the domain, accent, sentiment, duration, or keyword filters in the sidebar, "
                "or use Reset filters.")
    st.stop()

total_sec = calls_f["duration_sec"].sum()
kpi_row([
    ("Analyzed calls", f"{len(calls_f)}", f"of {len(ctx.calls)} in the study set"),
    ("Recorded min", f"{total_sec / 60:.1f}", format_duration(total_sec) + " of audio in total"),
    ("Avg duration", f"{calls_f['duration_min'].mean():.1f} min", "Mean recorded length per call"),
    ("Domains", f"{calls_f['domain'].nunique()}", f"{calls_f['accent'].nunique()} accent group(s)"),
    ("Turns scored", f"{len(ctx.sentiment_f)}",
     f"customer turns, of {len(ctx.turns_f)} transcript turns"),
])

left, right = st.columns([1, 1], gap="medium")
with left, card("ov_sent"):
    section_title("Overall call sentiment",
                  "One label per call (call-level grain), from the source sentiment analysis. "
                  "Not a satisfaction or CSAT score.")
    st.plotly_chart(charts.call_sentiment_distribution(calls_f), config=charts.CHART_CONFIG, theme=None,
                    key="ov_sent_chart", alt="Bar chart of call-level overall sentiment counts")
with right, card("ov_trans"):
    section_title("Opening → closing sentiment category",
                  "Count of calls by categorical opening and closing label. This is separate from the "
                  "numeric sentiment_change score.")
    st.plotly_chart(charts.transition_heatmap(calls_f, height=260), config=charts.CHART_CONFIG, theme=None,
                    key="ov_trans_chart", alt="Heatmap of opening versus closing sentiment categories")

with card("ov_mix"):
    head, ctrl = st.columns([3, 2], vertical_alignment="bottom")
    with head:
        section_title("Portfolio mix", "How the selected calls are distributed across business domains "
                                       "and English accent groups.")
    with ctrl:
        view = st.segmented_control("Mix view", ["Domains", "Accents", "Minutes by domain"],
                                    default="Domains", key="ov_mix_view", label_visibility="collapsed",
                                    required=True)
    if view == "Accents":
        fig = charts.category_mix(calls_f, "accent", height=240)
    elif view == "Minutes by domain":
        fig = charts.minutes_by_domain(calls_f)
    else:
        fig = charts.category_mix(calls_f, "domain")
    st.plotly_chart(fig, config=charts.CHART_CONFIG, theme=None, key="ov_mix_chart",
                    alt=f"Bar chart of {view.lower()} for the selected calls")

section_title("Call spotlight",
              "Calls ordered by call ID (the dataset contains no call dates). Open any call for the full "
              "summary, transcript, and sentiment timeline.")

SPOT_LIMIT = 6
show_all = len(calls_f) <= SPOT_LIMIT or st.toggle(f"Show all {len(calls_f)} matching calls", key="ov_show_all")
rows = calls_f if show_all else calls_f.head(SPOT_LIMIT)

cols = st.columns(3, gap="small")
for i, call in enumerate(rows.itertuples(index=False)):
    with cols[i % 3], card(f"spot_{call.call_id}", height="stretch"):
        st.html(
            f'<p class="wf-spot-id">{E(call.call_id)}</p>'
            f'<p class="wf-spot-title">{E(domain_label(call.domain))} · {E(accent_label(call.accent))}</p>'
            f'<p class="wf-spot-body">{E(call.customer_issue)}</p>'
            f'<div class="wf-spot-meta">{sentiment_pill(call.overall_sentiment)}'
            f'<span>{call.duration_min:.1f} min</span><span>·</span>'
            f'<span>{E(call.sentiment_transition)}</span></div>'
        )
        if st.button("Open in Call explorer", key=f"open_{call.call_id}", icon=":material/arrow_forward:",
                     width="stretch"):
            st.session_state["selected_call"] = call.call_id
            st.switch_page("app_pages/call_explorer.py")
