import pandas as pd
import streamlit as st

from components.ui import E, card, page_header, section_title
from utils.data_loader import load_calls, load_manifest, load_methodology, load_provenance

manifest = load_manifest()
method = load_methodology()
prov = load_provenance()
sources = prov.get("sources", {})
counts = manifest.get("counts", {})
calls = load_calls()
high_der = calls.loc[calls["diarization_der_pct"] >= 40, ["call_id", "diarization_der_pct"]]

page_header("About & methodology", "How these results were produced",
            "WeshFahmna AI is an offline explorer. Every transcript, summary, sentiment score, and evaluation "
            "metric was computed in a Google Colab study before this app existed. The app only reads and "
            "visualizes those stored results.")

STEPS = [
    ("Call audio", "12 English customer-service recordings. Audio files are referenced by path but not "
                   "included in this package.", "Not in package", "offline"),
    ("Speech recognition", f"{sources.get('asr', 'n/a')} transcribes each call.", "Offline study", ""),
    ("Diarization & roles", f"{sources.get('diarization', 'n/a')} segments speakers; agent/customer roles use "
                            f"{method.get('speaker_attribution', 'reference-assisted alignment')}.",
     "Reference-assisted", ""),
    ("Summarization", f"{sources.get('summary', 'n/a')} writes issue, action, outcome, and summary fields.",
     "Offline study", ""),
    ("Sentiment", f"{sources.get('sentiment', 'n/a')} scores customer turns with reference share ≥ "
                  f"{method.get('minimum_reference_share', 'n/a')}.", "Offline study", ""),
    ("Reviewed metrics", "WER, DER, and 1–5 rubric reviews of Qwen2.5-3B vs Qwen3-4B summaries.",
     "Offline study", ""),
    ("This dashboard", "Reads the stored files, applies filters, and draws charts. No models run here.",
     "App runtime", "runtime"),
]
steps_html = "".join(
    f'<div class="wf-step{" wf-step-" + cls if cls else ""}"><p class="wf-step-title">{E(t)}</p>'
    f'<p class="wf-step-body">{E(b)}</p><span class="wf-step-tag">{E(tag)}</span></div>'
    for t, b, tag, cls in STEPS
)
with card("ab_pipeline"):
    section_title("Pipeline", "Steps 1–6 happened in the offline study; only step 7 runs when you open this app.")
    st.html(f'<div class="wf-steps">{steps_html}</div>')

a, b = st.columns(2, gap="medium")
with a, card("ab_dims"):
    section_title("Dataset dimensions", "Read from app_data/data_manifest.json.")
    dims = pd.DataFrame({
        "Measure": ["Calls", "Transcript speaker turns", "Scored customer turns",
                    "Zero-length transcript turns", "Scored turns that diarization labelled agent",
                    "Transcript turns re-attributed to the reference role", "Mixed-speaker transcript segments",
                    "Mixed turns excluded from sentiment", "Manual checks of role attribution",
                    "Calls with partial closing-sentiment coverage", "Verified summary corrections"],
        "Value": [counts.get("calls"), counts.get("total_speaker_turns"), counts.get("scored_customer_turns"),
                  counts.get("nonpositive_duration_turns"),
                  counts.get("customer_turns_original_speaker_agent_but_reference_customer"),
                  counts.get("transcript_turns_role_corrected_to_reference"),
                  counts.get("transcript_turns_mixed_speaker"),
                  method.get("excluded_mixed_turns"), method.get("manual_checks"),
                  counts.get("calls_with_partial_closing_sentiment_coverage"), counts.get("verified_corrections")],
    })
    dims["Value"] = dims["Value"].map(lambda v: "n/a" if v is None else str(v))
    st.table(dims.set_index("Measure"))
with b, card("ab_models"):
    section_title("Source models", "Read from provenance.json and methodology.json.")
    models = pd.DataFrame({
        "Stage": ["Speech recognition", "Diarization", "Summarization", "Summarization baseline",
                  "Sentiment", "Speaker attribution"],
        "Model / method": [sources.get("asr"), sources.get("diarization"), sources.get("summary"),
                           "Qwen2.5-3B (comparison only)", sources.get("sentiment"),
                           method.get("speaker_attribution")],
    })
    models["Model / method"] = models["Model / method"].fillna("n/a")
    st.table(models.set_index("Stage"))

with card("ab_limits"):
    section_title("Limitations to keep in mind")
    limits = [
        f"Small sample: {len(calls)} calls across {calls['domain'].nunique()} domains and "
        f"{calls['accent'].nunique()} accent groups. Group-level differences are indicative only.",
        "ASR errors: transcripts and quoted evidence come from Whisper output and can contain mistakes.",
        "Diarization errors: "
        + ("; ".join(f"{r.call_id} has {r.diarization_der_pct:.1f}% DER" for r in high_der.itertuples()) + ". "
           if not high_der.empty else "")
        + "Speaker roles therefore rely on reference labels and are not a fully automated result.",
        "Sentiment model: a Twitter-trained RoBERTa model, not trained on call-center speech. Turn labels are model "
        "predictions, not human annotations, and are not a CSAT measure.",
        "Summary evaluation: 1–5 rubric scores reflect reviewer judgement on 12 examples, not a general benchmark.",
        f"Summary corrections: {counts.get('verified_corrections', 0)} Qwen3 summary fields were corrected after a "
        "targeted manual review against the transcript and are marked “verified correction”; the original model "
        "text is kept alongside. Other summary text was not exhaustively re-verified.",
        "No live inference, no audio, no call dates, agent IDs, verified resolution flags, or satisfaction surveys "
        "exist in this dataset — so none are shown.",
        *[f"Source note: {x}" for x in method.get("limitations", [])],
    ]
    if prov.get("limitations"):
        limits.append(f"Provenance note: {prov['limitations']}")
    st.markdown("\n".join(f"- {x}" for x in limits))

with card("ab_defs"):
    section_title("Metric definitions", "From the data manifest.")
    defs = manifest.get("definitions", {})
    st.table(pd.DataFrame({"Definition": list(defs.values())},
                          index=pd.Index([k.replace("_", " ").capitalize() for k in defs], name="Field")))
