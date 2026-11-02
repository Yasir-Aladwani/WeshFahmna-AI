import pandas as pd
import streamlit as st

from components.ui import (
    E, card, empty_state, page_header, scope_banner, section_title, sentiment_pill, sync_call_picker,
    text_block, transcript_html,
)
from utils import charts
from utils.context import get_context
from utils.data_loader import (
    accent_label, coverage_note, domain_label, format_duration, format_timestamp, load_corrections,
    load_provenance, load_source_table, load_summary_quality, speech_seconds_by_role,
)

DER_CAUTION_PCT = 40.0

ctx = get_context()

page_header("Call explorer", "Inspect any call end to end",
            "Browse the calls that match the sidebar filters, then open one to read its AI summary, "
            "the timestamped reference-assisted transcript, and its customer sentiment timeline.")
scope_banner(len(ctx.calls_f), len(ctx.calls), ctx.filters.is_active)

# ---------------------------------------------------------------- call table
with card("ex_table"):
    head, opt = st.columns([3, 1], vertical_alignment="bottom")
    with head:
        section_title("Calls", "Select a row to open it below. Sentiment change is the source numeric score "
                               "(not a percentage); Qwen3 quality is a reviewed rubric score out of 5.")
    with opt:
        show_metrics = st.toggle("Show WER / DER", key="ex_show_metrics",
                                 help="Add ASR word error rate and diarization error rate columns. "
                                      "Lower is better for both.")

    if ctx.calls_f.empty:
        empty_state("No calls match the current filters",
                    "Adjust or reset the sidebar filters, or turn on “Pick from all calls” below.")
    else:
        table = pd.DataFrame({
            "Call": ctx.calls_f["call_id"],
            "Domain": ctx.calls_f["domain"].map(domain_label),
            "Accent": ctx.calls_f["accent"].map(accent_label),
            "Duration (min)": ctx.calls_f["duration_min"],
            "Overall": ctx.calls_f["overall_sentiment"].str.title(),
            "Opening → closing": ctx.calls_f["sentiment_transition"].str.title(),
            "Sentiment change": ctx.calls_f["sentiment_change"],
            "Qwen3 quality": ctx.calls_f["qwen3_quality_out_of_5"],
        })
        if show_metrics:
            table["WER %"] = ctx.calls_f["asr_wer_pct"]
            table["Adjusted WER %"] = ctx.calls_f["asr_adjusted_wer_pct"]
            table["DER %"] = ctx.calls_f["diarization_der_pct"]
        table = table.reset_index(drop=True)
        st.session_state["_ex_table_ids"] = table["Call"].tolist()

        def _on_table_select():
            rows = st.session_state["ex_table"].selection.rows
            ids = st.session_state.get("_ex_table_ids", [])
            if rows and rows[0] < len(ids):
                st.session_state["selected_call"] = ids[rows[0]]

        st.dataframe(
            table, hide_index=True, key="ex_table", on_select=_on_table_select, selection_mode="single-row",
            alt="Table of calls matching the current filters",
            column_config={
                "Duration (min)": st.column_config.NumberColumn(format="%.1f"),
                "Sentiment change": st.column_config.NumberColumn(
                    format="%+.2f", help="Source-provided numeric change score; not a percentage or CSAT."),
                "Qwen3 quality": st.column_config.NumberColumn(
                    format="%.2f / 5",
                    help="Reviewed rubric score (1–5) for the Qwen3-4B summary. Not model confidence."),
                "WER %": st.column_config.NumberColumn(format="%.1f%%", help="Lower is better."),
                "Adjusted WER %": st.column_config.NumberColumn(format="%.1f%%", help="Lower is better."),
                "DER %": st.column_config.NumberColumn(format="%.1f%%", help="Lower is better."),
            },
        )

# ---------------------------------------------------------------- call picker
pick_all = st.toggle(f"Pick from all {len(ctx.calls)} calls (ignore filters)", key="ex_pick_all",
                     help="Lets you open any call even when the sidebar filters hide it.")
options = ctx.calls["call_id"].tolist() if pick_all else ctx.ids
previous = st.session_state.get("selected_call")
if not options:
    st.stop()
if previous and previous not in options:
    st.info(f"{previous} is hidden by the current filters, so the first matching call is shown instead. "
            "Turn on “Pick from all calls” to open it anyway.", icon=":material/filter_alt:")

call_id = sync_call_picker("Selected call", options, key="ex_call_picker")
call = ctx.calls.loc[ctx.calls["call_id"] == call_id].iloc[0]
turns = ctx.turns[ctx.turns["call_id"] == call_id]
sent = ctx.sentiment[ctx.sentiment["call_id"] == call_id]

# ---------------------------------------------------------------- detail header
st.html(f"""
<div class="wf-page-header">
  <p class="wf-eyebrow">{E(call_id)} · {E(domain_label(call.domain))} · {E(accent_label(call.accent))}</p>
  <div class="wf-spot-meta" style="margin-top:0">
    Overall {sentiment_pill(call.overall_sentiment)}
    <span>Opening</span>{sentiment_pill(call.opening_sentiment)}
    <span>→ Closing</span>{sentiment_pill(call.closing_sentiment)}
  </div>
</div>
""")

if call.diarization_der_pct >= DER_CAUTION_PCT:
    st.warning(f"Diarization error rate for this call is {call.diarization_der_pct:.1f}% (lower is better). "
               "Automated speaker segmentation was unreliable here; transcript roles shown are "
               "reference-assisted, but sentiment alignment and timings deserve extra caution.",
               icon=":material/warning:", title="Elevated diarization error")

if call.closing_sentiment_coverage == "partial":
    st.warning(coverage_note(call), icon=":material/timelapse:", title="Partial sentiment coverage")

speech = speech_seconds_by_role(turns)
meta = [
    ("Recorded duration", format_duration(call.duration_sec)),
    ("Transcript turns", f"{len(turns)} ({int(call.agent_turn_count)} agent · {int(call.customer_turn_count)} customer)"),
    ("Scored customer turns", f"{len(sent)}"),
    ("Numeric sentiment change", f"{call.sentiment_change:+.2f}"),
    ("Qwen3 summary quality", f"{call.qwen3_quality_out_of_5:.2f} / 5"),
]
st.html('<div class="wf-meta-grid">' + "".join(
    f'<div class="wf-meta"><p class="wf-meta-label">{E(k)}</p><p class="wf-meta-value">{E(v)}</p></div>'
    for k, v in meta) + "</div>")

tab_summary, tab_transcript, tab_timeline, tab_eval = st.tabs(
    ["Summary", "Transcript", "Sentiment timeline", "Evaluation & provenance"])

# ---------------------------------------------------------------- summary
with tab_summary:
    fixes = load_corrections()
    fixes = fixes[fixes["call_id"] == call_id].set_index("field")

    def summary_field(label: str, field: str) -> None:
        if field in fixes.index:
            text_block(f"{label} · verified correction", fixes.at[field, "corrected_text"], tone="verified")
        else:
            text_block(label, getattr(call, field))

    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        summary_field("Customer issue", "customer_issue")
    with c2:
        summary_field("Agent action", "agent_action")
    with c3:
        summary_field("Final outcome", "final_outcome")
    summary_field("Summary", "summary")
    text_block("Limitations / unverified details", call.notes_limitations, tone="caution")
    st.caption("Summary fields were generated by Qwen3-4B-Instruct-2507 during the offline study and are shown "
               "verbatim unless marked “verified correction”. The final outcome is the model’s reading of the "
               "transcript, not a verified resolution record.")
    if not fixes.empty:
        with st.expander(f"Verified corrections and original Qwen3 output ({len(fixes)} field(s))",
                         icon=":material/fact_check:"):
            st.caption("These fields were corrected after a manual review against the transcript. The original "
                       "model output is kept unchanged below for reproducibility, and the rubric scores on the "
                       "Evaluation tab refer to it.")
            for field, fix in fixes.iterrows():
                label = field.replace("_", " ").capitalize()
                text_block(f"{label} · why it was corrected · evidence at {fix.evidence_timestamps}", fix.issue,
                           tone="caution")
                text_block(f"{label} · original Qwen3 output", getattr(call, field))

    with card("ex_sent_expl"):
        section_title("Call-level sentiment rationale",
                      "Source justification and evidence strings, quoted as stored. Evidence is transcript "
                      "text and may contain ASR errors.")
        text_block("Justification", call.justification)
        e1, e2, e3 = st.columns(3, gap="small")
        with e1:
            text_block(f"Opening evidence · {call.opening_sentiment}", f"“{call.opening_evidence}”"
                       if call.opening_evidence else "")
        with e2:
            text_block(f"Closing evidence · {call.closing_sentiment}", f"“{call.closing_evidence}”"
                       if call.closing_evidence else "")
        with e3:
            text_block(f"Overall evidence · {call.overall_sentiment}", f"“{call.overall_evidence}”"
                       if call.overall_evidence else "")
        if pd.notna(call.sentiment_confidence):
            st.caption(f"Source call-level sentiment confidence: {call.sentiment_confidence:.2f} "
                       "(as stored by the sentiment step; not calibrated and not a satisfaction score).")

# ---------------------------------------------------------------- transcript
with tab_transcript:
    f1, f2 = st.columns([3, 2], vertical_alignment="bottom")
    with f1:
        query = st.text_input("Find in transcript", key=f"ex_find_{call_id}", icon=":material/search:",
                              placeholder="Highlight a word or phrase").strip()
    with f2:
        role_view = st.segmented_control("Speakers", ["All", "Agent", "Customer"], default="All",
                                         key="ex_role_view", required=True)
    shown = turns if role_view == "All" else turns[turns["role"] == role_view.lower()]
    n_hits = int(shown["text"].str.contains(query, case=False, regex=False).sum()) if query else 0
    zero = int((~turns["valid_time_span"]).sum())
    agent_starts = set(turns.loc[turns["role"] == "agent", "start"].round(2))
    conflicts = int(sent["start"].round(2).isin(agent_starts).sum())
    n_fixed = int(turns["role_corrected"].sum())
    n_mixed = int(turns["mixed_speaker_segment"].sum())
    st.caption(
        f"{len(shown)} turn(s) shown"
        + (f" · {n_hits} contain “{query}”" if query else "")
        + f" · speech time: agent {format_duration(speech.get('agent', 0))}, "
          f"customer {format_duration(speech.get('customer', 0))}"
        + (f" (excludes {zero} zero-length turn(s))" if zero else "")
        + ". Roles are reference-assisted (AppTek alignment, share ≥ 70%), not fully automated."
        + (f" {n_fixed} turn(s) differ from automated diarization and are marked “role corrected”." if n_fixed
           else "")
        + (f" {n_mixed} segment(s) contain speech from both parties and are marked “mixed speakers”." if n_mixed
           else "")
        + (f" {conflicts} agent-labelled turn(s) were attributed to the customer by the sentiment alignment "
           "and carry a role-conflict flag." if conflicts else "")
    )
    if shown.empty:
        empty_state("No turns to show", "This call has no transcript turns for the selected speaker.")
    else:
        with st.container(height=620, border=True):
            st.html(transcript_html(shown, sent, query))

# ---------------------------------------------------------------- sentiment timeline
with tab_timeline:
    if sent.empty:
        empty_state("No scored customer turns",
                    "This call has no customer turns that met the reference-share threshold for scoring.")
    else:
        with card("ex_timeline"):
            section_title("Customer sentiment over the call",
                          "Each point is one scored customer turn. Polarity = P(positive) − P(negative) from "
                          "the CardiffNLP RoBERTa model (range −1 to +1). Model output, not human review.")
            st.plotly_chart(charts.call_sentiment_timeline(sent), config=charts.CHART_CONFIG, theme=None,
                            key="ex_timeline_chart", alt=f"Customer sentiment polarity over time for {call_id}")
        lo = sent.loc[sent["sentiment_polarity"].idxmin()]
        hi = sent.loc[sent["sentiment_polarity"].idxmax()]
        a, b = st.columns(2, gap="small")
        with a:
            text_block(f"Lowest-polarity turn · {format_timestamp(lo.start)} · {lo.sentiment_polarity:+.2f}",
                       f"“{lo.text}”", tone="caution" if lo.sentiment_polarity < 0 else "default")
        with b:
            text_block(f"Highest-polarity turn · {format_timestamp(hi.start)} · {hi.sentiment_polarity:+.2f}",
                       f"“{hi.text}”")
        counts = sent["sentiment_label"].value_counts()
        st.caption(f"{len(sent)} scored turns in this call: {counts.get('positive', 0)} positive, "
                   f"{counts.get('neutral', 0)} neutral, {counts.get('negative', 0)} negative "
                   "(highest-probability label per turn).")
        if call.closing_sentiment_coverage == "partial":
            st.caption(coverage_note(call))

# ---------------------------------------------------------------- evaluation & provenance
with tab_eval:
    quality = load_summary_quality()
    q = quality[quality["call_id"] == call_id]
    dia = load_source_table("diarization_metrics.csv")
    d = dia[dia["call_id"] == call_id]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ASR WER", f"{call.asr_wer_pct:.1f}%", border=True,
              help="Whisper Small.en word error rate against the reference transcript. Lower is better.")
    m2.metric("Filler-adjusted WER", f"{call.asr_adjusted_wer_pct:.1f}%", border=True,
              help="WER after discounting filler words (um, uh…). Lower is better.")
    m3.metric("Diarization DER", f"{call.diarization_der_pct:.1f}%", border=True,
              help="Pyannote Community-1 diarization error rate. Lower is better.")
    m4.metric("Qwen3 summary quality", f"{call.qwen3_quality_out_of_5:.2f} / 5", border=True,
              help="Mean of reviewed rubric scores (factual accuracy, completeness, outcome accuracy), 1–5.")

    if not q.empty:
        r = q.iloc[0]
        rubric = pd.DataFrame({
            "Rubric (1–5)": ["Factual accuracy", "Completeness", "Outcome accuracy", "Overall (mean)"],
            "Qwen2.5-3B": [r.q25_factual, r.q25_completeness, r.q25_outcome, r.q25_overall],
            "Qwen3-4B": [r.q3_factual, r.q3_completeness, r.q3_outcome, r.q3_overall],
        })
        st.dataframe(rubric, hide_index=True, alt=f"Summary rubric scores for {call_id}",
                     column_config={"Qwen2.5-3B": st.column_config.NumberColumn(format="%.2f"),
                                    "Qwen3-4B": st.column_config.NumberColumn(format="%.2f")})
    prov = load_provenance().get("sources", {})
    details = {
        "Audio reference path (file not included)": call.audio_reference_path or "n/a",
        "Diarization inference time (s)": f"{d.iloc[0].inference_sec:.2f}" if not d.empty else "n/a",
        "Diarization aligned turns": f"{int(d.iloc[0].aligned_turns)}" if not d.empty else "n/a",
        **{f"Source model · {k.upper() if k == 'asr' else k.title()}": v for k, v in prov.items()},
    }
    st.table(pd.Series(details, name="Value").to_frame())
    st.caption("All metrics were computed during the offline study. Nothing on this page is recomputed live.")
