#!/usr/bin/env python3
"""Build offline, display-only Streamlit-ready views from the original integrated results.

Run from the project root: python scripts/prepare_data.py
No inference, no re-scoring, no external model/service required.

Requires the private original files in data/processed/integrated/ (not published).
Every free-text column written to app_data/ is passed through scripts/redaction.py,
so the generated views are publication-safe; numeric values are never altered.
"""
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from redaction import RESIDUAL_CHECKS, redact_columns, redact_turns

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'processed' / 'integrated'
OUTPUT = ROOT / 'app_data'
REFERENCE = OUTPUT / 'reference'

if not (SOURCE / 'calls.json').exists():
    sys.exit(f'Original integration files not found in {SOURCE}. They are kept private and are not part of '
             'the public repository; the published app_data/ views are already built.')
OUTPUT.mkdir(parents=True, exist_ok=True)

# Free-text columns that can contain personal identifiers, per output view.
TEXT_COLUMNS = {
    'calls_dashboard.csv': ['transcript', 'customer_issue', 'agent_action', 'final_outcome', 'summary',
                            'notes_limitations', 'justification', 'opening_evidence', 'closing_evidence',
                            'overall_evidence'],
    'transcript_turns_dashboard.csv': ['text'],
    'sentiment_timeline_dashboard.csv': ['text'],
}
# Non-personal source tables the app reads directly; copied verbatim.
REFERENCE_FILES = ['diarization_metrics.csv', 'final_quality_metrics.csv', 'methodology.json', 'provenance.json']
redaction_counts = Counter()


def load(name):
    return pd.read_csv(SOURCE / name)


def normalized_csv(df, filename, pre_redacted=False):
    cols = TEXT_COLUMNS.get(filename, [])
    out = df if pre_redacted else redact_columns(df, cols, redaction_counts)
    for col in cols:
        for check, pattern in RESIDUAL_CHECKS.items():
            leaked = out[col].astype(str).str.contains(pattern)
            if leaked.any():
                raise SystemExit(f'Redaction check "{check}" failed in {filename}:{col}; refusing to write.')
    out.to_csv(OUTPUT / filename, index=False, encoding='utf-8')


records = json.loads((SOURCE / 'calls.json').read_text(encoding='utf-8'))
source_calls = load('calls.csv')
source_turns = load('speaker_turns.csv')
source_sentiment = load('customer_turn_sentiment.csv')
source_quality = load('model_quality_comparison_12_calls.csv')
wer = load('filler_adjusted_wer.csv')

expected_ids = set(source_calls['call_id'])
assert len(records) == len(source_calls) == 12
assert {r['call_id'] for r in records} == expected_ids
assert set(source_turns['call_id']) == expected_ids
assert set(source_sentiment['call_id']) == expected_ids
assert source_calls['call_id'].is_unique
assert source_quality['call_id'].is_unique
assert source_sentiment[['neutral','positive','negative']].apply(pd.to_numeric).sum(axis=1).sub(1.0).abs().max() < 0.01

# Normalized one-row-per-call dataset. Raw text is preserved verbatim, not summarized again.
joined = source_calls.merge(wer[['call_id','adjusted_wer']], on='call_id', how='left', validate='one_to_one')
joined = joined.merge(source_quality[['call_id','q25_overall']], on='call_id', how='left', validate='one_to_one')
conf = pd.DataFrame([{
    'call_id': r['call_id'],
    'sentiment_confidence': r['sentiment'].get('confidence'),
    'audio_reference_path': r['metadata'].get('file_name'),
    'opening_evidence': r['sentiment'].get('opening_evidence'),
    'closing_evidence': r['sentiment'].get('closing_evidence'),
    'overall_evidence': r['sentiment'].get('overall_evidence'),
} for r in records])
joined = joined.merge(conf, on='call_id', how='left', validate='one_to_one')
joined['duration_min'] = joined['duration_sec'] / 60.0
joined['sentiment_transition'] = joined['opening_sentiment'] + ' → ' + joined['closing_sentiment']
joined['sentiment_delta_direction'] = joined['sentiment_change'].apply(
    lambda v: 'increasing' if v > 0 else ('decreasing' if v < 0 else 'unchanged')
)
joined['agent_turn_count'] = joined['call_id'].map(source_turns[source_turns['role']=='agent'].groupby('call_id').size()).fillna(0).astype(int)
joined['customer_turn_count'] = joined['call_id'].map(source_turns[source_turns['role']=='customer'].groupby('call_id').size()).fillna(0).astype(int)
joined['scored_customer_turn_count'] = joined['call_id'].map(source_sentiment.groupby('call_id').size()).fillna(0).astype(int)
joined['asr_wer_pct'] = joined['asr_wer'] * 100
joined['asr_adjusted_wer_pct'] = joined['adjusted_wer'] * 100
joined['diarization_der_pct'] = joined['diarization_der'] * 100
joined['qwen3_quality_out_of_5'] = joined['qwen3_quality']
joined['qwen25_quality_out_of_5'] = joined['q25_overall']
normalized_csv(joined, 'calls_dashboard.csv')

# Source-ordered speaker turns; retain zero-duration records for dialog, but do not treat as speech time.
turns = source_turns.sort_values(['call_id','start','end'],kind='stable').copy()
turns['duration_sec'] = (turns['end'] - turns['start']).clip(lower=0)
turns['valid_time_span'] = turns['end'] > turns['start']
# Turn-aware redaction: dictated numbers can be split across consecutive turns.
turn_key = list(zip(turns['call_id'], turns['start'].round(2)))
original_turn_text = dict(zip(turn_key, turns['text']))
turns['text'] = redact_turns(turns, redaction_counts)
redacted_turn_text = dict(zip(turn_key, turns['text']))
normalized_csv(turns, 'transcript_turns_dashboard.csv', pre_redacted=True)

# This subset uses reference-assisted role labels. Do not infer 'customer' from original_speaker.
sent = source_sentiment.sort_values(['call_id','start'],kind='stable').copy()
prob_cols = ['negative','neutral','positive']
sent['sentiment_label'] = sent[prob_cols].idxmax(axis=1)
sent['sentiment_class_confidence'] = sent[prob_cols].max(axis=1)
sent['sentiment_polarity'] = sent['positive'] - sent['negative']
sent['minute_in_call'] = sent['start'] / 60.0
# Scored turns carry the same text as their transcript turn; reuse its turn-aware redaction.
sent_key = list(zip(sent['call_id'], sent['start'].round(2)))
assert all(original_turn_text.get(k) == t for k, t in zip(sent_key, sent['text']))
sent['text'] = [redacted_turn_text[k] for k in sent_key]
normalized_csv(sent, 'sentiment_timeline_dashboard.csv', pre_redacted=True)

# Direct reproduction of the two compared Qwen quality scorecards, not a new evaluation.
quality = source_quality.merge(source_calls[['call_id','domain','accent']], on='call_id', how='left', validate='one_to_one')
normalized_csv(quality,'summary_quality_dashboard.csv')

# Keep provenance and methodological limits explicit, so the app can show them.
manifest = {
    'project': 'WeshFahmna AI',
    'type': 'offline_precomputed_demo',
    'counts': {
        'calls': int(len(joined)),
        'total_speaker_turns': int(len(turns)),
        'scored_customer_turns': int(len(sent)),
        'nonpositive_duration_turns': int((turns['end']<=turns['start']).sum()),
        'customer_turns_original_speaker_agent_but_reference_customer': int(((sent['original_speaker']=='Agent') & (sent['reference_role']=='customer')).sum()),
    },
    'original_data_root': 'data/processed/integrated (private; not published)',
    'reference_data_root': 'app_data/reference',
    'publication_safe': True,
    'redaction': {
        'method': 'rule-based masking of personal identifiers in free-text columns (scripts/redaction.py)',
        'placeholders': ['[EMAIL]', '[PHONE]', '[DATE OF BIRTH]', '[REFERENCE]', '[NUMBER]', '[CARD DIGITS]',
                         '[CODE]', '[SPELLED NAME]', '[SURNAME]'],
        'replacements_by_category': dict(sorted(redaction_counts.items())),
        'first_names_retained': True,
        'numeric_values_changed': False,
    },
    'prepared_views': ['calls_dashboard.csv','transcript_turns_dashboard.csv','sentiment_timeline_dashboard.csv','summary_quality_dashboard.csv'],
    'available_audio_files': False,
    'has_call_dates_or_times': False,
    'has_unique_agent_ids': False,
    'has_ground_truth_csat_or_resolution': False,
    'references_used_for_speaker_role_labels': True,
    'source_provenance': json.loads((SOURCE/'provenance.json').read_text(encoding='utf-8')),
    'source_methodology': json.loads((SOURCE/'methodology.json').read_text(encoding='utf-8')),
    'definitions': {
       'sentiment_polarity': 'positive_probability - negative_probability per scored customer turn (range -1 to 1)',
       'sentiment_class_confidence': 'highest of three probabilities for a scored customer turn; not calibrated call-wide confidence',
       'sentiment_delta_direction': 'sign of the supplied call-level sentiment_change score; not a categorical transition and not CSAT',
       'wer_der_percent': 'ratios multiplied by 100 for chart labels; lower is better',
       'qwen_quality': 'manual quality evaluation scored 1–5; NOT model confidence',
    },
}
REFERENCE.mkdir(parents=True, exist_ok=True)
for name in REFERENCE_FILES:
    shutil.copyfile(SOURCE / name, REFERENCE / name)

(OUTPUT/'data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print('Created dashboard data in', OUTPUT)
for f in sorted(p for p in OUTPUT.rglob('*') if p.is_file()):
    print(f.relative_to(OUTPUT).as_posix(), f.stat().st_size,'bytes')
print('Redactions by category:', dict(sorted(redaction_counts.items())))
