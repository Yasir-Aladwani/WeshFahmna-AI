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

from redaction import RESIDUAL_CHECKS, redact, redact_columns, redact_turns

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data' / 'processed' / 'integrated'
CORRECTIONS = ROOT / 'data' / 'corrections' / 'verified_corrections.json'
OUTPUT = ROOT / 'app_data'
REFERENCE = OUTPUT / 'reference'
# A segment whose words align to one reference speaker for less than this share is flagged as mixed.
MIXED_SEGMENT_BELOW = 0.9
# A call's closing sentiment counts as covered only if scored customer turns reach this far into the call.
CLOSING_COVERAGE_MIN_PCT = 90.0

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
    'verified_corrections.csv': ['issue', 'corrected_text'],
}
# Non-personal source tables the app reads directly; copied verbatim.
REFERENCE_FILES = ['diarization_metrics.csv', 'final_quality_metrics.csv', 'methodology.json', 'provenance.json']
redaction_counts = Counter()


def load(name):
    return pd.read_csv(SOURCE / name)


def mmss(seconds):
    return f'{int(seconds // 60)}:{int(seconds % 60):02d}'


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
methodology = json.loads((SOURCE / 'methodology.json').read_text(encoding='utf-8'))
min_share = float(methodology['minimum_reference_share'])
source_calls = load('calls.csv')
source_turns = load('speaker_turns.csv')
source_review = load('speaker_attribution_review.csv')
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

# Source-ordered speaker turns; retain zero-duration records for dialog, but do not treat as speech time.
turns = source_turns.sort_values(['call_id','start','end'],kind='stable').copy()
turns['duration_sec'] = (turns['end'] - turns['start']).clip(lower=0)
turns['valid_time_span'] = turns['end'] > turns['start']
turns['_key'] = turns['start'].round(2)
review = source_review.assign(_key=source_review['start'].round(2))[['call_id', '_key', 'reference_role',
                                                                    'reference_share']]
turns = turns.merge(review, on=['call_id', '_key'], how='left', validate='one_to_one')
assert turns['reference_role'].notna().all(), 'every transcript turn must have a reference alignment record'
# Speaker roles. The diarization role (Pyannote cluster mapping) is kept as diarization_role. The displayed
# role is the AppTek reference role wherever the segment aligns to one reference speaker for at least the
# study's minimum share, which is the same rule that selected the scored customer turns; below it the
# segment is too mixed to re-attribute, so the diarization role is kept and flagged as uncertain.
turns['diarization_role'] = turns['role']
turns['role_source'] = (turns['reference_share'] >= min_share).map({True: 'reference', False: 'diarization'})
turns['role'] = turns['reference_role'].where(turns['role_source'] == 'reference', turns['diarization_role'])
turns['role_corrected'] = turns['role'] != turns['diarization_role']
turns['mixed_speaker_segment'] = turns['reference_share'] < MIXED_SEGMENT_BELOW
turns = turns.drop(columns='_key')
# Turn-aware redaction: dictated numbers can be split across consecutive turns.
turn_key = list(zip(turns['call_id'], turns['start'].round(2)))
original_turn_text = dict(zip(turn_key, turns['text']))
turns['text'] = redact_turns(turns, redaction_counts)
redacted_turn_text = dict(zip(turn_key, turns['text']))
turn_role = dict(zip(turn_key, turns['role']))
turn_end = dict(zip(turn_key, turns['end']))
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
assert all(turn_role[k] == 'customer' for k in sent_key), 'scored turns must be customer turns in the transcript'
sent['text'] = [redacted_turn_text[k] for k in sent_key]
normalized_csv(sent, 'sentiment_timeline_dashboard.csv', pre_redacted=True)

# Normalized one-row-per-call dataset. Model text is kept verbatim (only redacted), never re-summarized.
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
by_call = turns.groupby('call_id')
joined['agent_turn_count'] = joined['call_id'].map(by_call['role'].apply(lambda r: int((r == 'agent').sum())))
joined['customer_turn_count'] = joined['call_id'].map(by_call['role'].apply(lambda r: int((r == 'customer').sum())))
joined['role_corrected_turn_count'] = joined['call_id'].map(by_call['role_corrected'].sum()).astype(int)
joined['mixed_speaker_turn_count'] = joined['call_id'].map(by_call['mixed_speaker_segment'].sum()).astype(int)
joined['scored_customer_turn_count'] = joined['call_id'].map(source_sentiment.groupby('call_id').size()).fillna(0).astype(int)
# How far into the call the scored customer turns reach, so partial closing coverage is not hidden.
sent['_end'] = [turn_end[k] for k in sent_key]
joined['first_scored_turn_sec'] = joined['call_id'].map(sent.groupby('call_id')['start'].min())
joined['last_scored_turn_end_sec'] = joined['call_id'].map(sent.groupby('call_id')['_end'].max())
joined['sentiment_coverage_end_pct'] = joined['last_scored_turn_end_sec'] / joined['duration_sec'] * 100
joined['closing_sentiment_coverage'] = joined['sentiment_coverage_end_pct'].ge(CLOSING_COVERAGE_MIN_PCT).map(
    {True: 'covered', False: 'partial'})
joined['asr_wer_pct'] = joined['asr_wer'] * 100
joined['asr_adjusted_wer_pct'] = joined['adjusted_wer'] * 100
joined['diarization_der_pct'] = joined['diarization_der'] * 100
joined['qwen3_quality_out_of_5'] = joined['qwen3_quality']
joined['qwen25_quality_out_of_5'] = joined['q25_overall']
joined = redact_columns(joined, TEXT_COLUMNS['calls_dashboard.csv'], redaction_counts)

# Verified corrections: published beside, never written over, the original model text.
spec = json.loads(CORRECTIONS.read_text(encoding='utf-8'))
turn_starts = turns.groupby('call_id')['start'].apply(list).to_dict()
corrections = []
for c in spec['corrections']:
    model_text = joined.loc[joined['call_id'] == c['call_id'], c['field']].iloc[0]
    if 'find' in c:
        assert model_text.count(c['find']) == 1, f"{c['call_id']}.{c['field']}: find text must occur once"
        corrected = model_text.replace(c['find'], c['replace'])
    else:
        corrected = c['corrected_text']
    assert redact(corrected) == corrected, f"{c['call_id']}.{c['field']}: correction contains an identifier"
    for s in c['evidence_turns']:
        assert any(abs(s - x) < 0.01 for x in turn_starts[c['call_id']]), f"{c['call_id']}: no turn at {s}s"
    corrections.append({
        'call_id': c['call_id'], 'field': c['field'], 'category': c['category'], 'issue': c['issue'],
        'corrected_text': corrected,
        'evidence_timestamps': ', '.join(mmss(s) for s in c['evidence_turns']),
        'evidence_turn_starts': ';'.join(f'{s:.2f}' for s in c['evidence_turns']),
        'review_method': spec['review_method'], 'review_date': spec['review_date'],
    })
corrections = pd.DataFrame(corrections)
assert not corrections.duplicated(['call_id', 'field']).any()
joined['verified_correction_fields'] = joined['call_id'].map(
    corrections.groupby('call_id')['field'].apply('; '.join)).fillna('')
normalized_csv(joined, 'calls_dashboard.csv', pre_redacted=True)
normalized_csv(corrections, 'verified_corrections.csv', pre_redacted=True)

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
        'transcript_turns_role_corrected_to_reference': int(turns['role_corrected'].sum()),
        'transcript_turns_mixed_speaker': int(turns['mixed_speaker_segment'].sum()),
        'transcript_turns_role_uncertain': int((turns['role_source'] == 'diarization').sum()),
        'scored_turns_not_customer_in_transcript': int(sum(turn_role[k] != 'customer' for k in sent_key)),
        'calls_with_partial_closing_sentiment_coverage': int((joined['closing_sentiment_coverage'] == 'partial').sum()),
        'verified_corrections': int(len(corrections)),
    },
    'speaker_roles': {
        'displayed_role': f'AppTek reference role where reference_share >= {min_share} (the rule used to select scored '
                          'customer turns); otherwise the diarization role, flagged as uncertain',
        'diarization_role': 'original Pyannote speaker-cluster role mapping, kept unchanged in diarization_role',
        'mixed_speaker_segment': f'reference_share < {MIXED_SEGMENT_BELOW}: the segment contains speech from both parties',
        'turn_counts_by_role': turns['role'].value_counts().to_dict(),
        'diarization_turn_counts_by_role': turns['diarization_role'].value_counts().to_dict(),
    },
    'sentiment_coverage': {
        'rule': f'closing sentiment coverage is "partial" when scored customer turns end before '
                f'{CLOSING_COVERAGE_MIN_PCT:.0f}% of the recorded call duration',
        'partial_calls': joined.loc[joined['closing_sentiment_coverage'] == 'partial', 'call_id'].tolist(),
    },
    'verified_corrections': {
        'file': 'verified_corrections.csv',
        'source': 'data/corrections/verified_corrections.json',
        'policy': 'Original Qwen3 summary fields stay verbatim in calls_dashboard.csv. Manually verified corrections '
                  'are published separately with transcript evidence; the app shows the corrected text and labels it.',
        'evaluation_scores_refer_to': 'the original model output (rubric scores are unchanged)',
        'by_call': corrections.groupby('call_id')['field'].apply(list).to_dict(),
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
    'prepared_views': ['calls_dashboard.csv','transcript_turns_dashboard.csv','sentiment_timeline_dashboard.csv',
                       'summary_quality_dashboard.csv','verified_corrections.csv'],
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
