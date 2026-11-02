# WeshFahmna AI — Prepared Streamlit Data

**Purpose:** A traceable, offline/demo-ready data layer built from the 12-call Maharah integration results. All numeric values and labels are preserved from the uploaded source; the `app_data/` views add only documented joins, display percentages, and straightforward chart fields. Free text in `app_data/` is publication-safe: personal identifiers are masked by `scripts/redaction.py` (see the README). **No models were rerun and no new AI-generated analyses were invented.**

## Folders

```text
WeshFahmna_Streamlit_Data_READY/
├── app_data/
│   ├── calls_dashboard.csv                 # 12 calls, filtered lists, headline KPIs, detail text
│   ├── transcript_turns_dashboard.csv      # 776 time-stamped agent/customer dialog turns
│   ├── sentiment_timeline_dashboard.csv    # 281 customer turns with three class probabilities
│   ├── summary_quality_dashboard.csv      # 12 Qwen2.5 vs Qwen3 evaluations
│   ├── verified_corrections.csv            # 9 manually verified summary-field corrections, with evidence
│   └── data_manifest.json                  # counts, provenance, limits, definitions
├── data/corrections/verified_corrections.json  # correction source (published; no personal data)
├── data/processed/integrated/              # all 11 ORIGINAL source files, unchanged (PRIVATE, git-ignored)
│   ├── calls.csv
│   ├── calls.json
│   ├── customer_turn_sentiment.csv
│   ├── speaker_turns.csv
│   ├── speaker_attribution_review.csv
│   ├── diarization_metrics.csv
│   ├── filler_adjusted_wer.csv
│   ├── model_quality_comparison_12_calls.csv
│   ├── final_quality_metrics.csv
│   ├── methodology.json
│   └── provenance.json
├── scripts/prepare_data.py                 # rebuilds app_data from original files
├── DATA_REVIEW.md                          # audit and dashboard specification
└── CLAUDE_CODE_IMPLEMENTATION_PROMPT.md    # paste into Claude Code
```

## What is actually available

- **12** calls, **8** domains, **4** English-accent groups.
- **776** agent/customer transcript turns, each with start and end seconds and text.
  - **Displayed roles:** 443 agent and 333 customer. These are reference-assisted: the AppTek role is used where reference share ≥ 0.7.
  - **Diarization roles:** 424 agent and 352 customer, kept in `diarization_role`.
  - **Flags:** `role_corrected` (85 turns) and `mixed_speaker_segment` (208 turns, reference share < 0.9).
- **281** scored customer sentiment turns containing `neutral`, `positive`, `negative` probabilities. **These are turn-level probabilities, not call-level CSAT.**
- Call summaries: `customer_issue`, `agent_action`, `final_outcome`, full narrative summary, `notes_limitations`. These are original Qwen3 output, verbatim apart from redaction. Nine fields in CALL_005, CALL_010, CALL_011 and CALL_012 have manually verified corrections in `verified_corrections.csv`, with transcript evidence timestamps; the model text itself is not edited.
- Sentiment coverage per call: `first_scored_turn_sec`, `last_scored_turn_end_sec`, `sentiment_coverage_end_pct`, and `closing_sentiment_coverage`. That last field is `partial` when the scored turns end before 90% of the call; only CALL_006 is partial, at 71%.
- Call-level sentiment labels, opening/closing labels, numeric `sentiment_change`, justifications, confidence/evidence strings.
- Quality evaluations: Whisper Small.en original/filler-adjusted WER, Pyannote diarization DER, Qwen2.5-3B and Qwen3-4B **human-style rubric scores on a 1–5 scale**.

## Accuracy and representation restrictions

1. The uploaded ZIP contains **NO `.wav`/`.mp3` audio files**. `metadata.file_name` is a reference path, not a playable file. Do not add a broken audio player.
2. There are **NO call calendar dates**, agent employee IDs, CSAT survey scores, verified resolution labels, or ticket/SLA data. Do not invent time-series by day/month, leaderboards, CSAT rates, or resolution percentages.
3. Customer-vs-agent speaker roles and sentiment inclusion are **reference-assisted**, using AppTek labels and a threshold of `reference_share >= 0.7`. **Not equivalent to a validated fully automated production pipeline**. `speaker_attribution_review.csv` includes additional raw review records; do not use `original_speaker` in the sentiment CSV to override the corrected `reference_role`. The transcript view applies the same rule, so all 281 scored turns are customer turns in the transcript. Segments below the threshold keep their diarization role and are marked as uncertain rather than relabelled.
4. **Seven** transcript turns have equal start and end timestamps. Keep the words in the transcript, but exclude these zero-length records from any duration-derived calculations.
5. `sentiment_change` is a stored **numeric score**; it is not a categorical transition. A neutral-to-neutral call can still have positive numeric change. Separate these two visuals clearly.
6. `WER` and `DER` use decimal fractions (e.g. `0.20` = 20%). The prepared dataset provides explicit `_pct` display columns. **Lower is better**.
7. Qwen summary-quality evaluation is on a **1–5 rubric**, not a live model confidence; the comparison is from these twelve reviewed examples, not a universal benchmark.
8. Review all transcripts, personal information, permissions, and source data licensing **before committing the data to a public GitHub repository or deploying a public Streamlit app**.

## Preparing data again

Run from the extracted project folder:

```bash
python -m pip install pandas
python scripts/prepare_data.py
```

This regenerates the derived `app_data/` CSVs and manifest from the original unchanged data. A future Streamlit app will additionally need `streamlit`, `plotly`, and `pandas`.

## Design intent

A professional **English-only** product/demo titled **WeshFahmna AI**, with an international SaaS aesthetic, interactive call exploration, accessible charts, and an honest evaluation methodology. It is a precomputed-results explorer — **not a live call-processing engine**.
