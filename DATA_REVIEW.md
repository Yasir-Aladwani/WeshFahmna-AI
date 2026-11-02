# WeshFahmna AI — Data Audit and Visualization Plan

Audit of `Maharah_Integration_Results_FINAL(1).zip`, performed on 2026-11-02. Every original file was retained without modification.

## 1. Inventory

| Original file | Rows / structure | Main use |
| --- | ---: | --- |
| `calls.json` | 12 nested records | canonical full record, evidence, full transcript, metadata, per-call nested evaluations |
| `calls.csv` | 12 × 19 | call details, summary, sentiments, WER/DER, Qwen3 quality |
| `speaker_turns.csv` | 776 × 6 | timestamped transcript, reference-assisted role labels |
| `customer_turn_sentiment.csv` | 281 × 9 | customer turn sentiment probabilities and position in call |
| `speaker_attribution_review.csv` | 776 × 6 | role-attribution methodology audit, raw vs reference labels |
| `diarization_metrics.csv` | 12 × 6 | DER, inference seconds, aligned turns, domain/accent |
| `filler_adjusted_wer.csv` | 12 × 4 | ASR WER before/after filler adjustment |
| `model_quality_comparison_12_calls.csv` | 12 × 10 | rubric breakdown of Qwen2.5 versus Qwen3 |
| `final_quality_metrics.csv` | 4 × 4 | aggregate quality scorecards for models |
| `methodology.json` | 8 fields | model names, attribution method, exclusion criteria, limitations |
| `provenance.json` | 4 fields | source models, row counts, caveats |

All 12 call IDs match across main call, transcript, sentiment, and evaluation tables. No missing cells in the principal original CSV files. Customer sentiment probability columns sum to approximately 1 per row.

## 2. Observed dataset profile

| Measure | Actual value |
| --- | ---: |
| Total calls | 12 |
| Total call minutes | 118.83 (1h 58m 50s) |
| Average call length | 9.90 min |
| Domains | 8 |
| Accent groups | 4 |
| Call-level sentiment | 4 positive, 8 neutral, 0 negative |
| Opening → closing categorical transition | 6 neutral→positive; 3 neutral→neutral; 2 positive→neutral; 1 positive→positive |
| Supplied numeric sentiment change | 9 positive change, 3 negative change |
| Reference-assisted speaker turns | 776 (424 agent / 352 customer) |
| Customer turns with sentiment scores | 281 |
| Turn-level highest-probability labels | 181 neutral, 79 positive, 21 negative |
| Average original WER | 20.85% |
| Average filler-adjusted WER | 18.24% |
| Average DER | 19.93% |
| Qwen2.5 mean overall quality | 3.31 / 5 |
| Qwen3 mean overall quality | 3.83 / 5 |
| Mean Qwen3 minus Qwen2.5 | +0.53 points / 5 |

**Interpretation warnings:** The 281 turn-level sentiment labels must not be merged into the 12 call-level label distribution. Likewise, a positive numerical sentiment-change score is not identical to a category moving from neutral to positive. A zero count of negative *overall* calls does not imply all interactions were positive or that customers were satisfied.

## 3. Recommended visual elements

**Executive Overview:** total calls; call minutes; average duration; domain count; call-level overall sentiment split; domain/accent mix; opening-to-closing sentiment transitions; a call preview list.

**Call Explorer:** searchable call list; filters for domain, accent, sentiment, length; each detail shows customer issue, agent action, final outcome, summary/limitations, score explanation and evidence; complete timestamped reference-assisted speaker transcript; colored customer-turn sentiment markers and sentiment timeline. Do not require audio playback.

**Customer Sentiment:** distribution of the 281 scored customer turns; polarity (`positive_probability - negative_probability`) along elapsed seconds/minutes for a selected call; sentiment transition table; call-level numeric sentiment-change ranking; turn-level tooltip with original text and score breakdown.

**Model Quality & Methodology:** original vs filler-adjusted WER per call; DER per call/accent; Qwen2.5 vs Qwen3 quality per call and by rubric (factual accuracy, completeness, outcome accuracy); clear annotation of high-DER/WER examples; source models and manual/reference-assisted limitations. Display quality as study metadata, not as call agent performance.

## 4. Findings / guardrails

- The package has no attached audio recordings, call calendar dates, agent IDs, ground-truth resolution status, CSAT, FCR, AHT operational instrumentation, or SLA evidence. Any such dashboard values would be **fabricated**. Note: average *recording* duration is available, but not instrumented handle time.
- `speaker_attribution_review.csv` and `customer_turn_sentiment.csv` contain original model speaker tags that may differ from the aligned reference roles. In the sentiment file, **32** rows have original speaker `Agent` but a reference-assisted role of `customer`. Display corrected reference roles and disclose the dependency. This is not a model-free real-time classifier.
- **7** out of 776 timed transcript turns have zero duration (`end == start`); keep their text while excluding from speech-duration calculations.
- Relatively high diarization DER on `CALL_005` (55.0%) and `CALL_006` (50.9%) warrants visible quality cautions if those call transcripts are viewed. High error metrics reduce confidence in raw automated speaker attribution.
- Qwen3 improves the overall mean quality, but **4** of the twelve call-specific comparisons favor Qwen2.5, and **1** ties. Avoid declaring Qwen3 best in all calls.
- All records are **English-language examples**; do not create artificial Arabic localization or Saudi-themed visual treatment.

## 5. Added app-ready data and their exact definitions

| File | Grain / fields | Transformations |
| --- | --- | --- |
| `app_data/calls_dashboard.csv` | one row/call | original call data + WER adjusted/Qwen2.5 joins, JSON sentiment confidence/evidence, duration in minutes, transition string, numeric score direction, role-turn counts, percentage display columns |
| `app_data/transcript_turns_dashboard.csv` | one row/speaker turn | sort by call+start; `valid_time_span`; `duration_sec = max(end-start, 0)` |
| `app_data/sentiment_timeline_dashboard.csv` | one scored customer turn | `sentiment_label = argmax(negative, neutral, positive)`; max class probability; polarity=`positive - negative`; minutes=`start/60` |
| `app_data/summary_quality_dashboard.csv` | one row/call | raw Qwen scorecard + domain/accent descriptions |
| `app_data/data_manifest.json` | dataset metadata | counts, model provenance, explicit methodological and data limitations |

Run `python scripts/prepare_data.py` to regenerate all views. No new synthetic records, sentiment model scoring, re-transcription, or summarization were performed.
