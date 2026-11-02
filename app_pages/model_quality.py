import pandas as pd
import streamlit as st

from components.ui import card, empty_state, kpi_row, page_header, scope_banner, section_title
from utils import charts
from utils.context import get_context
from utils.data_loader import load_provenance, load_source_table, load_summary_quality

DER_CAUTION_PCT = 40.0

ctx = get_context()
sources = load_provenance().get("sources", {})

page_header("Model quality", "How well did each pipeline stage perform?",
            "Study-level evaluation of the offline pipeline: ASR by "
            f"{sources.get('asr', 'n/a')}, diarization by {sources.get('diarization', 'n/a')}, "
            f"summarization by {sources.get('summary', 'n/a')} (compared with Qwen2.5-3B), and sentiment by "
            f"{sources.get('sentiment', 'n/a')}. Speaker roles use reference-assisted alignment (AppTek). "
            "These are model evaluation metrics — they say nothing about agent performance.")
scope_banner(len(ctx.calls_f), len(ctx.calls), ctx.filters.is_active,
             extra="Means below are computed over the calls in scope.")

if ctx.calls_f.empty:
    empty_state("No calls match the current filters", "Widen or reset the sidebar filters to see evaluation "
                                                      "results.")
    st.stop()

calls_f = ctx.calls_f
quality = load_summary_quality()
quality_f = quality[quality["call_id"].isin(ctx.ids)]
wins = int((quality_f["improvement"] > 0).sum())
losses = int((quality_f["improvement"] < 0).sum())
ties = int((quality_f["improvement"] == 0).sum())

kpi_row([
    ("Mean original WER", f"{calls_f['asr_wer_pct'].mean():.2f}%", "Whisper Small.en · lower is better"),
    ("Mean filler-adjusted WER", f"{calls_f['asr_adjusted_wer_pct'].mean():.2f}%",
     "Fillers (um, uh…) discounted · lower is better"),
    ("Mean DER", f"{calls_f['diarization_der_pct'].mean():.2f}%", "Pyannote Community-1 · lower is better"),
    ("Summary quality", f"{quality_f['q25_overall'].mean():.2f} → {quality_f['q3_overall'].mean():.2f}",
     "Qwen2.5-3B → Qwen3-4B, mean out of 5"),
])

tab_asr, tab_dia, tab_sum = st.tabs(["Speech recognition (ASR)", "Diarization", "Summarization"])

with tab_asr:
    with card("mq_wer"):
        section_title("Word error rate by call",
                      "WER = (substitutions + deletions + insertions) ÷ reference words, shown as a percentage. "
                      "Filler-adjusted WER discounts hesitation words that the reference may omit. "
                      "Lower is better.")
        st.plotly_chart(charts.wer_by_call(calls_f), config=charts.CHART_CONFIG, theme=None, key="mq_wer_chart",
                        alt="Grouped bars of original and filler-adjusted WER per call")
        best = calls_f.loc[calls_f["asr_wer_pct"].idxmin()]
        worst = calls_f.loc[calls_f["asr_wer_pct"].idxmax()]
        st.caption(f"Lowest original WER: {best.call_id} ({best.asr_wer_pct:.1f}%). Highest: {worst.call_id} "
                   f"({worst.asr_wer_pct:.1f}%). Evaluated against reference transcripts on this 12-call set only; "
                   "results may not generalize to other audio conditions.")

with tab_dia:
    high = calls_f[calls_f["diarization_der_pct"] >= DER_CAUTION_PCT]
    if not high.empty:
        st.warning("; ".join(f"{r.call_id}: {r.diarization_der_pct:.1f}% DER" for r in high.itertuples())
                   + ". Automated speaker segmentation was unreliable on these calls; this is one reason "
                     "transcript roles use reference-assisted alignment.",
                   icon=":material/warning:", title="Elevated diarization error")
    a, b = st.columns([3, 2], gap="medium")
    with a, card("mq_der"):
        section_title("Diarization error rate by call",
                      "DER = share of audio time with missed speech, false alarms, or speaker confusion. "
                      f"Bars at or above {DER_CAUTION_PCT:.0f}% are highlighted. Lower is better.")
        st.plotly_chart(charts.der_by_call(calls_f, DER_CAUTION_PCT), config=charts.CHART_CONFIG, theme=None,
                        key="mq_der_chart", alt="Bar chart of DER per call")
    with b, card("mq_der_accent"):
        section_title("Mean DER by accent group", "Small groups (2–4 calls each); treat differences as "
                                                   "indicative, not conclusive.")
        st.plotly_chart(charts.der_by_accent(calls_f), config=charts.CHART_CONFIG, theme=None,
                        key="mq_der_acc_chart", alt="Bar chart of mean DER by accent group")
        dia = load_source_table("diarization_metrics.csv")
        dia_f = dia[dia["call_id"].isin(ctx.ids)]
        st.caption(f"Mean diarization inference time: {dia_f['inference_sec'].mean():.1f} s per call "
                   "(measured offline in Colab, not in this app).")

with tab_sum:
    st.caption("Summaries were scored by a reviewer on a 1–5 rubric for factual accuracy, completeness, and "
               "outcome accuracy; overall is their mean. These are evaluation scores, not model confidence, and "
               "reviewer judgement is subjective.")
    a, b = st.columns([3, 2], gap="medium")
    with a, card("mq_qwen_call"):
        section_title("Overall quality by call", "Both models summarized the same 12 calls.")
        st.plotly_chart(charts.qwen_by_call(quality_f), config=charts.CHART_CONFIG, theme=None,
                        key="mq_qwen_chart", alt="Grouped bars of Qwen2.5 and Qwen3 overall quality per call")
    with b, card("mq_rubric"):
        section_title("Rubric scorecard", "Mean per rubric over the calls in scope.")
        rubric = pd.DataFrame({
            "metric": ["Factual accuracy", "Completeness", "Outcome accuracy", "Overall"],
            "Qwen2.5-3B": [quality_f[c].mean() for c in ["q25_factual", "q25_completeness", "q25_outcome",
                                                         "q25_overall"]],
            "Qwen3-4B": [quality_f[c].mean() for c in ["q3_factual", "q3_completeness", "q3_outcome",
                                                       "q3_overall"]],
        })
        st.plotly_chart(charts.rubric_scorecard(rubric), config=charts.CHART_CONFIG, theme=None,
                        key="mq_rubric_chart", alt="Grouped bars of mean rubric scores by model")
    with card("mq_delta"):
        section_title("Where Qwen3 improved — and where it did not",
                      f"Qwen3-4B scored higher on {wins} call(s), lower on {losses}, and tied on {ties}. "
                      "Cyan = Qwen3 higher, violet = Qwen2.5 higher.")
        st.plotly_chart(charts.qwen_delta_by_call(quality_f, height=max(240, 32 * len(quality_f) + 90)),
                        config=charts.CHART_CONFIG, theme=None, key="mq_delta_chart",
                        alt="Diverging bars of per-call score difference between Qwen3 and Qwen2.5")
    with st.expander("Full-study scorecard as stored in final_quality_metrics.csv"):
        st.dataframe(load_source_table("final_quality_metrics.csv"), hide_index=True,
                     alt="Aggregate quality metrics for all 12 calls",
                     column_config={"metric": "Metric",
                                    "improvement": st.column_config.NumberColumn("Improvement", format="%+.2f")})
        st.caption("Source aggregate over all 12 calls, independent of the sidebar filters.")
