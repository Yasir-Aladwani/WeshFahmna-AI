# WeshFahmna AI

**Voice Intelligence • Call Analytics** — an interactive Streamlit explorer for a 12-call English customer-service study. It shows transcripts, reference-assisted speaker turns, AI call summaries, customer sentiment, and model-evaluation results.

> This is an **offline demo over precomputed results**. All transcription, diarization, summarization, sentiment scoring, and evaluation were done earlier in a Google Colab study. The app only reads the stored files and charts them. It runs no models, makes no API calls, and includes no audio.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501. The app needs Python 3.10 or later and has been tested on Python 3.12 and 3.14 with Streamlit 1.66, pandas 3.1, and Plotly 6.9.

The app reads only the publication-safe files in `app_data/`; nothing else is needed to run it.

### Deploying to Streamlit Community Cloud

1. Push the repository to GitHub. The private originals are git-ignored and are never needed.
2. In Streamlit Community Cloud, create an app from the repository with **main file `app.py`** and Python 3.12.
3. No secrets or environment variables are required. Dependencies install from `requirements.txt`, and the theme loads from `.streamlit/config.toml`.

## Dataset attribution and licence

The conversations analysed here come from:

- **Dataset:** AppTek Call-Center Dialogues
- **Source:** https://huggingface.co/datasets/apptek-com/apptek_callcenter_dialogues
- **Licence:** [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)

This project uses 12 dialogues from that dataset. Its reference speaker labels were used for the reference-assisted speaker-role alignment.

The files in `app_data/` are **adapted material** derived from the dataset; they are modified as described below. Under the ShareAlike term, they are distributed under **CC BY-SA 4.0**. The dataset authors do not endorse this project. The application source code is separate from the data and is not covered by this data licence.

## Data transformations

Every result was produced offline in a Google Colab study. This repository adds no new model outputs.

1. **Offline study (not in this repository).** The 12 audio recordings went through Whisper Small.en transcription and Pyannote Community-1 diarization. Speaker roles were then aligned with AppTek reference labels. Qwen3-4B-Instruct-2507 wrote the summaries, with Qwen2.5-3B as a comparison. Customer turns were scored with `cardiffnlp/twitter-roberta-base-sentiment-latest`. Reviewers rated the summaries on a 1–5 rubric. The study exported 11 integration files, which are **kept private**, git-ignored and unmodified.
2. **Display views (`scripts/prepare_data.py`).** This script joins the call, WER and quality tables. It derives durations in minutes, `_pct` display columns, the opening → closing transition string, and per-call turn counts. For each scored turn it adds the argmax sentiment label, the highest class probability and the polarity (P(positive) − P(negative)). It also sorts the turns by time. No values are re-scored or re-estimated.
3. **Publication-safe redaction (`scripts/redaction.py`).** This step runs automatically inside `prepare_data.py` before any file is written; see the next section.
4. **Speaker roles and sentiment coverage (`scripts/prepare_data.py`).** Each transcript turn shows the AppTek reference role when at least 70% of the segment aligns to one reference speaker; otherwise it keeps the diarization role and is marked uncertain. This is the same rule the study used to choose scored customer turns. The original diarization role is kept in `diarization_role`. The script also records how far into each call the scored turns reach. See [Data accuracy review](#data-accuracy-review).
5. **Verified corrections (`data/corrections/verified_corrections.json`).** Manually checked fixes to Qwen3 summary fields, each with transcript timestamps as evidence. They are published as `app_data/verified_corrections.csv` and **never overwrite** the original model text in `calls_dashboard.csv`.
6. **Reference copies.** Four non-personal source tables (`diarization_metrics.csv`, `final_quality_metrics.csv`, `methodology.json`, `provenance.json`) are copied verbatim into `app_data/reference/`.

You only need to rebuild the views if you hold the private originals in `data/processed/integrated/`:

```bash
python scripts/prepare_data.py
```

## Publication-safe redactions

Personal identifiers are masked in every free-text field the app shows. This covers the full call transcript, transcript turns, scored sentiment turns, the customer issue, agent action, final outcome, summary, limitations, justification, and evidence quotes. Masking uses ordered, rule-based placeholders:

| Placeholder | What it replaces |
| --- | --- |
| `[EMAIL]` | Email addresses, whether written ("x@domain.com"), spoken ("x at gmail dot com", "at the rate of"), or spelled letter by letter, including addresses split across transcript turns |
| `[PHONE]` | Phone numbers in digits, dashed or spaced form |
| `[NUMBER]` | Dictated digit strings, such as phone, security-card, loyalty-card and booking numbers. These include sequences split across consecutive turns and numbers given in answer to a prompt like "phone number?" or "read the code" |
| `[CARD DIGITS]`, `[CODE]` | The last four digits of a payment card, and a one-time verification code |
| `[REFERENCE]` | Long purchase or reference numbers |
| `[DATE OF BIRTH]` | Full dates of birth |
| `[SURNAME]` | Surnames of customers, agents and third parties. **First names are kept** so the dialogue stays readable |
| `[SPELLED NAME]` | Names spelled out letter by letter |

What the redaction does **not** change:
- **Numeric values, labels, timestamps and row counts.** All metrics (WER, DER, sentiment probabilities, rubric scores) are identical to the unredacted build; this was verified column by column.
- **Prices, amounts, plan names and times**, such as "$600", "349 plan" or "11 15 a.m.".
- **Keyword search on masked terms.** Searching for a masked identifier no longer matches, by design.

`prepare_data.py` refuses to write a view if any residual email, phone or surname pattern survives. `data_manifest.json` records the redaction method and the number of replacements per category; it never stores the original values. The source is a publicly released dataset, but these identifiers are masked anyway as a precaution.

## Data accuracy review

An independent review of the published data found the issues below. Each one was checked against the time-stamped transcript and the AppTek reference alignment. Model outputs and evaluation metrics were **not** changed: WER, DER, sentiment probabilities and rubric scores are identical to the earlier build, and the rubric scores refer to the original Qwen3 text.

**Speaker roles.** For 32 scored turns (14 in CALL_005, 7 in CALL_006), the transcript said "agent" while the sentiment data said "customer". The transcript role came from Pyannote diarization, which is unreliable in these calls (DER 55.0% and 50.9%). The sentiment role came from the AppTek reference alignment. The transcript now uses the reference role where the alignment is decisive (share ≥ 70%). That re-attributes 85 of 776 turns in both directions: the 33 high-share turns, which include the reviewer's 32, plus 52 customer→agent turns. Every scored turn now appears as a customer turn. Mixed segments are not relabelled blindly:
- **Mixed segments.** 208 segments align less than 90% to one speaker, so they contain speech from both parties. The app marks them "mixed speakers".
- **Uncertain segments.** The 108 segments below 70% keep their diarization role and are marked "role uncertain".
- **Diarization role kept.** The original role is still available in `diarization_role`.

**Summary corrections.** Nine fields across four calls are shown as "verified correction" in Call explorer, with the original Qwen3 output and the evidence timestamps one click away.

| Call | Fields | Correction |
| --- | --- | --- |
| CALL_012 | agent action, final outcome, summary | The total is **$830** ($630 food + $150 barista and coffee machine + $50 cleanup), not $780. $780 was the subtotal before cleanup was added. The payment timing is stated cautiously because that part of the transcript is garbled. |
| CALL_010 | agent action, final outcome, summary | The customer asked for $600 but agreed to **$500** because of the $500 daily limit; the limit was not "adjusted". The agent confirmed completion without restating the amount, and the customer did not confirm the balance. |
| CALL_005 | final outcome, summary | The items were **reserved or added to the basket** for in-store inspection. Final payment was not made or confirmed, so this was not a completed purchase. |
| CALL_011 | summary | The granny flat is for the customer's **partner's mother** ("my partner's mom", "my mum-in-law"), not their own mother. |

**Sentiment coverage.** CALL_006 has only 8 scored customer turns, from 2:19 to 7:06 of a 10:03 call (71%). Later customer speech sits in mixed segments that were never scored. The app now flags this as "partial" closing coverage wherever opening and closing sentiment are shown. A call is flagged when its scored turns end before 90% of the call; no other call is below 95%.

**Redaction fix.** The written-email rule swallowed the first word of the following sentence ("[EMAIL] items"). The rule now stops at a dot that is followed by a space.

## Application screens

| Screen | What it shows |
| --- | --- |
| **Overview** | KPI tiles that follow the filters: analyzed calls, recorded minutes, average recorded duration, domains, scored customer turns. Also the call-level sentiment split, opening → closing category heatmap, domain/accent/minutes mix, and a call spotlight ordered by call ID that links into Call explorer. |
| **Call explorer** | A filterable call table where you can select a row (WER/DER columns are optional). Each call has a detail view with customer issue, agent action, final outcome, summary, and limitations. Verified corrections are labelled, and the original Qwen3 text is shown beside them. It also has a chat-style timestamped transcript with safe search highlighting, a customer sentiment timeline, and per-call evaluation and provenance. |
| **Sentiment insights** | Call-level labels (12) and turn-level predictions (281), kept separate. Covers opening vs closing categories, the numeric `sentiment_change` by call, the share of predicted turn labels, a turn-polarity strip, and a timeline for one selected call. |
| **Model quality** | Original vs filler-adjusted WER, DER per call and by accent (CALL_005/006 flagged), and Qwen2.5-3B vs Qwen3-4B rubric scores. Shows per-call wins **and** losses. |
| **About & methodology** | Pipeline stepper (offline study vs app runtime), dataset dimensions and source models read from the manifest/provenance files, limitations, and metric definitions. |

The sidebar filters (keyword, domain, accent, overall sentiment, duration band) apply to the first four screens. Each screen has a banner saying whether it shows all 12 calls or a filtered subset.

## Project layout

```text
app.py                     # entry point: page config, sidebar filters, st.navigation
app_pages/                 # one script per screen
  overview.py  call_explorer.py  sentiment.py  model_quality.py  about.py
components/ui.py           # HTML helpers (hero, KPI tiles, transcript); all data is html-escaped
utils/data_loader.py       # cached CSV/JSON loading, filter logic, formatting helpers
utils/context.py           # applies the global filters for each page run
utils/charts.py            # Plotly figures with one shared visual system
styles/app.css             # styling for app-owned .wf-* classes
assets/                    # SVG logo and icon
.streamlit/config.toml     # dark theme
tests/test_app.py          # headless acceptance checks (streamlit.testing AppTest)
app_data/                  # PUBLISHED, publication-safe display views (CC BY-SA 4.0)
  reference/               # verbatim copies of 4 non-personal source tables
scripts/prepare_data.py    # rebuilds app_data/ from the private originals and applies redaction
scripts/redaction.py       # masking rules and leak checks
data/corrections/          # manually verified summary corrections with transcript evidence (published)
data/processed/integrated/ # PRIVATE: 11 original integration files, git-ignored, never modified
```

## Data and evaluation methodology

- **12 calls**, 8 domains, 4 English accent groups; **776** transcript turns, of which 7 have zero-length timestamps. These 7 stay visible in the transcript but are left out of speech-time totals.
- **281 scored customer turns.** Each has three class probabilities from `cardiffnlp/twitter-roberta-base-sentiment-latest`. The turn label is the class with the highest probability. Polarity = P(positive) − P(negative).
- **Speaker roles are reference-assisted** (AppTek alignment, reference share ≥ 0.7), not fully automated. Transcript and sentiment now use the same rule. 85 turns differ from automated diarization and are flagged "role corrected", and 208 mixed-speaker segments are flagged. With reference roles, the transcript has 443 agent and 333 customer turns; diarization alone gave 424 and 352.
- **Sentiment coverage.** CALL_006's scored turns stop at 71% of the call, so its closing label is flagged as partial.
- **Summaries** are Qwen3 output shown verbatim, except 9 manually verified corrections, which are labelled as such. Other summary text was not exhaustively re-verified.
- **`sentiment_change`** is a numeric score from the source data, not a percentage and not CSAT. The opening → closing *category* transition is a separate field, so the two are charted separately.
- **WER/DER** are error rates, so lower is better. Whisper Small.en has a mean original WER of 20.85% and a mean filler-adjusted WER of 18.24%. Pyannote Community-1 has a mean DER of 19.93%, but CALL_005 (55.0%) and CALL_006 (50.9%) are much higher.
- **Summary quality** is a reviewer's score on a 1–5 rubric (factual accuracy, completeness, outcome accuracy), not model confidence. Qwen3-4B averages 3.83/5 against 3.31/5 for Qwen2.5-3B. Qwen3 scored lower on 4 calls and tied on 1.
- The dataset has **no** call dates, agent IDs, CSAT, resolution flags, or SLA/AHT data, so the app shows none of these.

## Running the checks

```bash
pip install pytest
python -m pytest tests -q
```

The checks confirm the following:

- `prepare_data.py` leaves the 11 originals unchanged (SHA-256 compared before and after). This test is skipped in a public clone, where the originals are absent.
- The published `app_data/` text contains no residual email, phone or surname patterns.
- The redaction rules behave as expected on synthetic examples, and the app's runtime code never reads the private folder.
- The dataset counts are correct.
- Every scored sentiment turn is a customer turn in the transcript; roles change only where the reference alignment is decisive, and the diarization roles are preserved.
- Verified corrections cite transcript turns that exist, the original model text is unchanged, and CALL_006 is flagged for partial sentiment coverage.
- Every page renders without exceptions.
- The unfiltered KPIs are correct.
- CALL_001 to CALL_012 can each be selected.
- Filters update the views, and an empty result shows a proper empty state.
- A selected call that a filter removes falls back cleanly.
- Keyword search reaches transcript turns.
- HTML in transcript text is escaped.

## Publication and privacy note

- **Only redacted data is published.** `app_data/` contains the masked views described above. The unredacted integration results in `data/processed/integrated/` are git-ignored. Never force-add them, and never upload the original ZIP export.
- If you regenerate the views, rerun `python -m pytest tests -q` before you commit. The leak check fails if an identifier slips through.
- Rule-based masking is not a guarantee. If you add new transcripts, review the output by hand before you publish.
- No credentials or secrets are needed or stored. `.streamlit/secrets.toml` is git-ignored.
