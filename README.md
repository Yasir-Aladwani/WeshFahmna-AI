<h1 align="center">WeshFahmna AI | وش فهمنا؟</h1>

<p align="center">
  <strong>AI-Powered Call Center Conversation Intelligence</strong><br>
  Turning customer-service conversations into structured summaries, sentiment insights, and auditable model-quality results.
</p>

<p align="center">
  <img src="assets/branding/weshfahmna-cover.svg" alt="WeshFahmna AI — Call Center Conversation Intelligence" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Speech-Whisper-087E8B?style=flat-square" alt="Whisper ASR">
  <img src="https://img.shields.io/badge/LLM-Qwen-7759C2?style=flat-square" alt="Qwen LLMs">
  <img src="https://img.shields.io/badge/Dataset-CC_BY--SA_4.0-2A8A79?style=flat-square" alt="Dataset license">
</p>

<p align="center">
  <a href="#overview">Overview</a> ·
  <a href="#system-architecture">Architecture</a> ·
  <a href="#evaluation-results">Evaluation</a> ·
  <a href="#run-locally">Run Locally</a> ·
  <a href="#data-license--attribution">Data & License</a>
</p>

Project scope: This is a 12-call evaluation and visualization prototype, not a live call-monitoring or employee-scoring system. Speech processing and language-model inference are performed offline; the Streamlit dashboard reads prepared results. The original dataset consists of role-played English customer-service calls, not private customer recordings.

Table of Contents

#

Section

Description

01

Overview

What WeshFahmna AI does

02

Project Problem

Why conversation analysis matters

03

Key Features

Main analytical capabilities

04

System Architecture

Offline processing and dashboard layers

05

AI Pipeline

From recordings to actionable insights

06

Dataset & Scope

Public data source and pilot coverage

07

Models & Tech Stack

Speech, diarization, NLP, and visualization

08

Interactive Dashboard

What the application shows

09

Evaluation Results

Measured quality and sample statistics

10

Reproducibility

How the results are organized

11

Run Locally

Start the Streamlit interface

12

Repository Structure

Code, data, and documentation layout

13

Limitations

What the system does not establish

14

Data License & Attribution

Dataset credit and permitted uses

15

Author

Project owner

Overview

WeshFahmna AI | وش فهمنا؟ is an end-to-end AI project that explores how customer-service conversations can be transformed into understandable, reviewable insights.

The pipeline combines automatic speech recognition, speaker diarization, structured call summarization, and customer sentiment analysis. Its outputs are joined into a Streamlit dashboard for examining individual calls, overall conversation trends, and model reliability.

Instead of presenting AI predictions without context, the project also reports Word Error Rate (WER), Diarization Error Rate (DER), model-comparison scores, and data-provenance notes. This makes the difference between what the system inferred and what the reference data supports explicit.

At a glance

12 calls analyzed

118.83 minutes

776 speaker turns

281 customer turns scored

8 service domains

4 accent groups

2 summarization models compared

4 main dashboard views

Project Problem

Long customer-service calls contain useful information, but reviewing them individually is time-consuming. Critical details can be scattered across many turns: why the customer called, what the agent did, what outcome was reached, and how the customer's language changed during the interaction.

A useful analysis tool should organize these details while also exposing uncertainty. Errors in transcription, speaker attribution, and generated summaries can affect downstream interpretation. WeshFahmna AI therefore treats model quality and traceability as part of the product, not an afterthought.

Key Features

Capability

What it provides

Call Summarization

Structured customer issue, agent action, and outcome/next-step fields from Qwen models

Speech Transcription

English ASR outputs generated with faster-whisper (small.en)

Speaker Analysis

Diarization experiments, timestamped turns, and reviewable attribution results

Customer Sentiment

Call-level sentiment, customer-turn classification, and beginning-to-end comparisons

Interactive Exploration

Searchable calls, filters, transcripts, evidence, and sentiment timelines

Quality Evaluation

Per-call WER/DER, adjusted WER, and Qwen2.5-versus-Qwen3 quality comparisons

Reproducible Outputs

Structured CSV/JSON exports, methodology notes, and provenance metadata

System Architecture

The system separates compute-intensive offline inference from the lightweight, read-only Streamlit interface. This is intentional: dashboard visitors can inspect results without downloading large models or reprocessing audio.

flowchart TB
    A["AppTek Call-Center Dialogues<br/>public benchmark audio + references"]
    subgraph P["Offline AI Processing"]
        B["Speech recognition<br/>faster-whisper small.en"]
        C["Speaker diarization<br/>pyannote.audio"]
        D["Call summarization<br/>Qwen2.5 / Qwen3"]
        E["Customer sentiment<br/>RoBERTa"]
        F["Evaluation + source alignment<br/>WER, DER, rubric, attribution audit"]
    end
    subgraph S["Prepared Data Layer"]
        G["Validated CSV / JSON exports"]
        H["Dashboard-ready data views"]
    end
    subgraph U["Presentation Layer"]
        I["Streamlit dashboard"]
        J["Overview · Call Explorer<br/>Sentiment · Model Quality"]
    end
    A --> B
    A --> C
    A --> D
    A --> E
    A --> F
    B --> F
    C --> F
    D --> F
    E --> F
    F --> G --> H --> I --> J

    classDef core fill:#0d253b,stroke:#18aeb6,color:#eefaff
    classDef data fill:#163045,stroke:#8cbfe4,color:#eefaff
    classDef ui fill:#253b50,stroke:#b9a1ff,color:#eefaff
    class B,C,D,E,F core
    class G,H data
    class I,J ui

Speaker-label disclosure: The dashboard's role-labeled transcripts use reference-assisted speaker attribution where available. Raw diarization is evaluated separately and should not be mistaken for fully automatic, error-free agent/customer identification.

<!-- Optional: after uploading your previously designed system architecture SVG,
     add it here with the correct relative path, for example:
<p align="center"><img src="assets/architecture/system_architecture.svg" alt="System Architecture" width="95%"></p>
-->

AI Pipeline

Select & inspect calls from the public AppTek diarization configuration.

Transcribe speech using faster-whisper small.en and compare predictions with reference text.

Analyze speakers using pyannote; retain both diarization metrics and reference-assisted role labels for the dashboard.

Summarize each call into the customer issue, agent action, and outcome or next step; compare Qwen2.5 and Qwen3 outputs.

Score customer sentiment using a RoBERTa classifier; aggregate call-level interpretations and turn-level sentiment trajectories.

Evaluate and audit WER, filler-adjusted WER, DER, summary quality, and attribution limitations.

Prepare reproducible exports and serve the read-only analytical experience through Streamlit.

flowchart LR
    A["Public call audio"] --> B["ASR + speaker analysis"]
    B --> C["Summaries + sentiment"]
    C --> D["Evaluation + provenance"]
    D --> E["CSV / JSON"]
    E --> F["Streamlit insights"]
    style A fill:#0d253b,stroke:#18aeb6,color:#ffffff
    style F fill:#253b50,stroke:#b9a1ff,color:#ffffff

<!-- Optional: replace the Mermaid pipeline with your exported Project Pipeline SVG
     once it is committed to assets/pipeline/. -->

Dataset & Scope

Source: AppTek Call-Center Dialogues by AppTek.ai.

This benchmark contains role-played, long-form English customer-service conversations with accent, domain, transcript, and speaker annotations. The project uses its diarization configuration and analyzes a 12-call pilot sample (selection seed: 42).

Pilot characteristic

Value

Calls

12

Total recording duration

118.83 min

Average recording duration

9.90 min

Service domains represented

8

Accent groups represented

4

Timestamped speaker turns

776

Customer turns with sentiment scores

281

The pilot is not the entire AppTek benchmark. The dataset's broader range of accents and service domains should not be confused with the coverage of this 12-call experiment.

Models & Tech Stack

Layer

Technology / model

Purpose

ASR

faster-whisper · small.en

English speech transcription

Speaker diarization

pyannote.audio

Automatic speaker segmentation and DER evaluation

Summarization

Qwen2.5-3B-Instruct

Comparison baseline

Summarization

Qwen3-4B-Instruct-2507

Structured summary generation and comparative evaluation

Sentiment

cardiffnlp/twitter-roberta-base-sentiment-latest

Customer utterance sentiment classification

Data analysis

Python · pandas · NumPy

Joining, validation, aggregations, metrics

Visualization

Streamlit · Plotly

Interactive exploration and analytics

Experimentation

Google Colab

Offline model experimentation

Formats

CSV · JSON

Interoperable, reviewable outputs

Important: A general-domain sentiment model can misinterpret customer-service language; its predictions are analytical signals, not verified customer satisfaction scores.

Interactive Dashboard

The interface is designed around four complementary views:

View

Main questions answered

Executive Overview

How many calls were analyzed? How are domains, accents, recording lengths, and overall sentiments distributed?

Call Explorer

What happened in a selected call? What do the summary, transcript, customer sentiment, and quality notes show?

Customer Sentiment

Which turns were classified positive/neutral/negative? How did the sentiment signal change within a call?

Model Quality & Methodology

How reliable are transcription, diarization, and summaries? What role do references and manual review play?

Screenshots

The layout is prepared for the final Streamlit screenshots. Once screenshots are committed, add them using this pattern:

<!--
### Executive Overview
<p align="center"><img src="assets/screenshots/overview.png" alt="Executive Overview" width="95%"></p>

### Call Explorer
<p align="center"><img src="assets/screenshots/call_explorer.png" alt="Call Explorer" width="95%"></p>

### Sentiment Analysis
<p align="center"><img src="assets/screenshots/customer_sentiment.png" alt="Customer Sentiment" width="95%"></p>

### Model Quality
<p align="center"><img src="assets/screenshots/model_quality.png" alt="Model Quality" width="95%"></p>
-->

Evaluation Results

All results below refer to the 12-call pilot, using the project's stored evaluation tables—not the full AppTek benchmark.

Speech and speaker quality

Metric

Pilot mean

Interpretation

Original WER

20.85%

Average word error rate before filler adjustment; lower is better

Filler-adjusted WER

18.24%

Adjusted comparison after defined filler handling; lower is better

DER

19.93%

Average speaker diarization error rate; lower is better

WER improved by 2.61 percentage points under the project's filler-adjusted calculation. This adjusted metric must not be reported as directly interchangeable with standard WER.

Summarization quality comparison

Model

Mean rubric score

Qwen2.5

3.31 / 5

Qwen3

3.83 / 5

Mean difference (Qwen3 − Qwen2.5)

+0.53 / 5

The project uses a structured review rubric covering aspects such as factual accuracy, completeness, and outcome accuracy. Qwen3 scored higher on average, but Qwen2.5 performed better on 4 of the 12 calls, with 1 tie. These are small-sample review scores, not population-level guarantees.

Customer sentiment snapshot

Unit of analysis

Distribution

Overall call labels (12)

8 neutral · 4 positive · 0 negative

Customer-turn labels (281)

181 neutral · 79 positive · 21 negative

Opening → closing call labels

6 neutral→positive · 3 neutral→neutral · 2 positive→neutral · 1 positive→positive

The call-level and turn-level distributions are different measurements and must not be combined. A non-negative overall call label is not proof of satisfaction, resolution, or excellent agent performance.

Reproducibility

The processing stage produces auditable outputs rather than making new model calls on each dashboard visit.

Export

Contents

calls.json / calls.csv

Per-call metadata, summaries, and headline metrics

speaker_turns.csv

Timestamped, role-labeled transcript turns

customer_turn_sentiment.csv

Customer-turn sentiment scores and probabilities

diarization_metrics.csv

Per-call DER and alignment statistics

filler_adjusted_wer.csv

Standard versus adjusted ASR comparison

model_quality_comparison_12_calls.csv

Qwen2.5 and Qwen3 rubric comparisons

final_quality_metrics.csv

Aggregated model evaluations

speaker_attribution_review.csv

Audit trail for reference-assisted roles

methodology.json / provenance.json

Sources, model versions, assumptions, and caveats

For the Streamlit app, scripts/prepare_data.py assembles the analytical views in app_data/, including call summaries, transcript turns, sentiment timelines, model quality, and a dataset manifest.

No inference is rerun when viewing precomputed dashboard pages. Full model-level reproduction requires the original source dataset, model dependencies, processing notebooks/scripts, and the documented experiment settings.

Run Locally

The commands below assume the repository uses app.py as its Streamlit entry point and includes requirements.txt and the prepared app_data/ folder.

git clone https://github.com/Yasir-Aladwani/WeshFahmna-AI.git
cd WeshFahmna-AI
python -m venv .venv

Windows (PowerShell):

.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py

macOS / Linux:

source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py

If you have the original integration exports and need to refresh the dashboard views:

python scripts/prepare_data.py

Deployment note: Streamlit Community Cloud should load the prepared dashboard files. The heavy ASR, diarization, and LLM experiments are designed to run separately, not on each app startup.

Repository Structure

A clean repository organization for the processing outputs and dashboard is:

WeshFahmna-AI/
├── app.py                      # Streamlit entry point
├── app_data/                   # Dashboard-ready CSV/JSON views
├── scripts/
│   └── prepare_data.py        # Rebuild analytical views
├── data/
│   └── processed/              # Audited integration exports (when included)
├── notebooks/                  # Offline experiments (when included)
├── assets/
│   ├── branding/
│   │   └── weshfahmna-cover.svg
│   ├── architecture/           # Add exported architecture SVG here
│   ├── pipeline/               # Add exported pipeline SVG here
│   └── screenshots/            # Add final app screenshots here
├── requirements.txt
└── README.md

This is the target layout for publication; update filenames in this section if the final GitHub repository differs.

Limitations

Small pilot: 12 role-played calls do not represent all accents, domains, or real-world service conditions.

Reference-assisted roles: dashboard customer/agent labels depend on dataset references; raw diarization is a separately evaluated experiment. Some calls have high DER (for example, CALL_005 and CALL_006).

ASR and summaries are imperfect: recognition mistakes can propagate into generated summaries and sentiment interpretation.

Sentiment is not CSAT: inferred tone is not a customer survey, and it cannot establish whether an issue was resolved.

No operational KPIs: the available data does not establish true Average Handle Time, First Contact Resolution, SLA compliance, or agent productivity. Recording duration is not the same as instrumented handle time.

Not a real-time application: the published dashboard is a visualization of preprocessed outputs, not a live audio ingestion service.

Human review remains necessary: model outputs should not be used as the sole basis for evaluating employees or making consequential decisions.

Data License & Attribution

The underlying benchmark is AppTek Call-Center Dialogues, created and shared by AppTek.ai, under Creative Commons Attribution–ShareAlike 4.0 (CC BY-SA 4.0).

Dataset: AppTek Call-Center Dialogues on Hugging Face

License terms: CC BY-SA 4.0

Source dataset use: The benchmark is provided for evaluation and analysis, not for training or fine-tuning ASR/foundation models.

Attribution: Credit AppTek.ai when sharing original or adapted dataset content. Applicable share-alike obligations may apply to redistributed adaptations.

This repository documents the project's own processing, evaluation, and presentation. The dataset license is not automatically a license declaration for independently authored source code.

Author

Yasir Aladwani — AI Engineer · Data Science & Machine Learning

LinkedIn · GitHub

<p align="center"><sub>WeshFahmna AI — From conversations to clarity, with evidence and evaluation.</sub></p>
