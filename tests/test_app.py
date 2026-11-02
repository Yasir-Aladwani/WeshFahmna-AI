"""Headless acceptance checks. Run: python -m pytest tests -q  (or python tests/test_app.py)."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from components.ui import highlight, transcript_html  # noqa: E402
from redaction import RESIDUAL_CHECKS, redact, redact_turns  # noqa: E402
from utils.data_loader import DURATION_BANDS, CallFilters, filter_calls  # noqa: E402

PAGES = ["app_pages/overview.py", "app_pages/call_explorer.py", "app_pages/sentiment.py",
         "app_pages/model_quality.py", "app_pages/about.py"]
SOURCE = ROOT / "data" / "processed" / "integrated"


def _app(page: str | None = None, **state) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if page:
        at.switch_page(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _html(at: AppTest) -> str:
    return "\n".join(str(getattr(el, "proto", el)) for el in at.get("html"))


def _raw():
    calls = pd.read_csv(ROOT / "app_data" / "calls_dashboard.csv")
    turns = pd.read_csv(ROOT / "app_data" / "transcript_turns_dashboard.csv")
    sent = pd.read_csv(ROOT / "app_data" / "sentiment_timeline_dashboard.csv")
    return calls, turns, sent


@pytest.mark.skipif(not (SOURCE / "calls.json").exists(),
                    reason="private original integration files are not part of the public repository")
def test_prepare_data_does_not_mutate_sources():
    files = sorted(p for p in SOURCE.iterdir() if p.is_file())
    assert len(files) == 11
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py")], check=True, cwd=ROOT,
                   capture_output=True)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    assert before == after


def test_dataset_counts():
    calls, turns, sent = _raw()
    assert len(calls) == 12 and len(turns) == 776 and len(sent) == 281
    assert calls["domain"].nunique() == 8 and calls["accent"].nunique() == 4
    assert calls["overall_sentiment"].value_counts().to_dict() == {"neutral": 8, "positive": 4}
    assert int((turns["end"] <= turns["start"]).sum()) == 7


def test_all_pages_render():
    for page in PAGES:
        _app(page)


def test_unfiltered_overview_kpis():
    html = _html(_app())
    for expected in ["Analyzed calls", ">12<", "Turns scored", ">281<", "of 776 transcript turns",
                     ">8<", "All 12 calls"]:
        assert expected in html, expected


def test_every_call_selectable():
    calls, _, _ = _raw()
    for call_id in calls["call_id"]:
        at = _app("app_pages/call_explorer.py", selected_call=call_id)
        assert at.selectbox(key="ex_call_picker").value == call_id


def test_filters_update_and_empty_state():
    at = _app(f_domains=["food"])
    html = _html(at)
    assert "2 of 12 calls" in html and ">2<" in html
    at = _app(f_keyword="zzzz-no-such-phrase")
    assert "No calls match the current filters" in _html(at)
    for page in PAGES[1:4]:
        _app(page, f_keyword="zzzz-no-such-phrase")


def test_duration_bands_partition_calls():
    calls, turns, _ = _raw()
    for col in ["customer_issue", "summary", "transcript"]:
        calls[col] = calls[col].fillna("")
    sizes = [len(filter_calls(calls, turns, CallFilters(duration_bands=(b,)))) for b in DURATION_BANDS]
    assert sum(sizes) == 12 and all(sizes)
    html = _html(_app(f_duration=["12 min or more"]))
    assert f"{sizes[-1]} of 12 calls" in html


def test_filtered_away_selection_falls_back():
    at = _app("app_pages/call_explorer.py", selected_call="CALL_005", f_domains=["food"])
    assert at.selectbox(key="ex_call_picker").value in {"CALL_001", "CALL_012"}
    assert any("hidden by the current filters" in i.value for i in at.info)


def test_keyword_filter_searches_transcript_turns():
    calls, turns, _ = _raw()
    for col in ["customer_issue", "summary", "transcript"]:
        calls[col] = calls[col].fillna("")
    turns["text"] = turns["text"].fillna("")
    hit = filter_calls(calls, turns, CallFilters(keyword="vegan"))
    assert "CALL_001" in set(hit["call_id"])


def test_html_escaping():
    payload = '<script>alert(1)</script><img src=x onerror=alert(2)>'
    out = highlight(payload, "script")
    assert "<script>" not in out and "<img" not in out and "<mark>script</mark>" in out
    turns = pd.DataFrame({"call_id": ["X"], "start": [0.0], "end": [0.0], "role": ["customer"],
                          "text": [payload], "valid_time_span": [False]})
    out = transcript_html(turns, pd.DataFrame(), query="&amp")
    assert "<script>" not in out and "zero-length timestamp" in out


def test_published_data_has_no_personal_identifiers():
    published = sorted(p for p in (ROOT / "app_data").rglob("*") if p.suffix in {".csv", ".json"})
    assert published
    for path in published:
        if path.suffix == ".csv":
            df = pd.read_csv(path)
            text = "\n".join(df[c].astype(str).str.cat(sep="\n") for c in df.columns if df[c].dtype == object
                             or str(df[c].dtype) == "str")
        else:
            text = path.read_text(encoding="utf-8")
        for name, pattern in RESIDUAL_CHECKS.items():
            assert not pattern.search(text), f"{name} identifier found in {path.relative_to(ROOT)}"
    manifest = (ROOT / "app_data" / "data_manifest.json").read_text(encoding="utf-8")
    assert '"publication_safe": true' in manifest


def test_redaction_rules_on_synthetic_text():
    # Synthetic values only; no real identifiers belong in this repository.
    cases = {
        "mail me at jane.doe@example.com today": "[EMAIL]",
        "it's sam dot lee at gmail.com right": "[EMAIL]",
        "call 555 -010 -0199 please": "[PHONE]",
        "born 3rd of May 1990 in town": "[DATE OF BIRTH]",
        "the last four digits of the card? Yeah, 1234.": "[CARD DIGITS]",
        "Thank you Miss Black, and Mr. Smith": "[SURNAME]",
    }
    for raw, token in cases.items():
        assert token in redact(raw), raw
    assert redact("black and gold heels for $600") == "black and gold heels for $600"
    assert redact("sent to sam@example.com. All items") == "sent to [EMAIL]. All items"
    turns = pd.DataFrame({"call_id": ["X"] * 3,
                          "text": ["Can I take a phone number?", "four five", "three two okay"]})
    assert redact_turns(turns, Counter())[1:] == ["[NUMBER]", "[NUMBER] okay"]


def test_transcript_roles_agree_with_sentiment_and_keep_diarization():
    _, turns, sent = _raw()
    role = dict(zip(zip(turns["call_id"], turns["start"].round(2)), turns["role"]))
    assert all(role[k] == "customer" for k in zip(sent["call_id"], sent["start"].round(2)))
    # The original diarization roles are preserved, and roles change only where the reference is decisive.
    assert turns["diarization_role"].value_counts().to_dict() == {"agent": 424, "customer": 352}
    changed = turns[turns["role"] != turns["diarization_role"]]
    assert changed["role_corrected"].all() and (changed["role_source"] == "reference").all()
    assert (changed["reference_share"] >= 0.7).all() and (changed["role"] == changed["reference_role"]).all()
    assert (turns.loc[turns["role_source"] == "diarization", "reference_share"] < 0.7).all()
    assert turns.loc[turns["reference_share"] < 0.9, "mixed_speaker_segment"].all()


def test_verified_corrections_keep_model_output():
    calls, turns, _ = _raw()
    fixes = pd.read_csv(ROOT / "app_data" / "verified_corrections.csv")
    fix = {(r.call_id, r.field): r.corrected_text for r in fixes.itertuples()}
    model = calls.set_index("call_id")
    # Original model output stays verbatim; corrections are separate.
    assert "$780" in model.at["CALL_012", "summary"] and "$780" in model.at["CALL_012", "final_outcome"]
    assert "purchased" in model.at["CALL_005", "summary"]
    for field in ["summary", "final_outcome"]:
        assert "$830" in fix[("CALL_012", field)] and "total of $780" not in fix[("CALL_012", field)]
        assert "$500" in fix[("CALL_010", field)] and "did not restate" in fix[("CALL_010", "final_outcome")]
        assert "purchased" not in fix[("CALL_005", field)] and "not made or confirmed" in fix[("CALL_005", field)]
    assert "$500 to $500" not in fix[("CALL_010", "agent_action")]
    assert "partner's mother" in fix[("CALL_011", "summary")]
    starts = turns.groupby("call_id")["start"].apply(lambda s: set(s.round(2)))
    for r in fixes.itertuples():
        assert r.field in calls.columns and r.issue and r.evidence_timestamps
        assert {round(float(x), 2) for x in r.evidence_turn_starts.split(";")} <= starts[r.call_id]
    assert set(calls.loc[calls["verified_correction_fields"].notna(), "call_id"]) == set(fixes["call_id"])


def test_partial_sentiment_coverage_is_flagged():
    calls, _, _ = _raw()
    partial = calls[calls["closing_sentiment_coverage"] == "partial"]
    assert partial["call_id"].tolist() == ["CALL_006"]
    assert 420 < partial.iloc[0]["last_scored_turn_end_sec"] < 430
    at = _app("app_pages/call_explorer.py", selected_call="CALL_006")
    assert any("Partial sentiment coverage" in str(w.proto) for w in at.warning)
    assert "Partial closing coverage: CALL_006" in "".join(c.value for c in _app("app_pages/sentiment.py").caption)


def test_call_explorer_shows_corrections_and_role_flags():
    at = _app("app_pages/call_explorer.py", selected_call="CALL_012")
    html = _html(at)
    assert "verified correction" in html and "$830" in html and "original Qwen3 output" in html
    at = _app("app_pages/call_explorer.py", selected_call="CALL_005")
    html = _html(at)
    assert "role corrected" in html and "mixed speakers" in html and "role conflict" not in html


def test_runtime_never_reads_private_originals():
    for path in [ROOT / "app.py", *(ROOT / "app_pages").glob("*.py"), *(ROOT / "utils").glob("*.py"),
                 *(ROOT / "components").glob("*.py")]:
        assert "processed" not in path.read_text(encoding="utf-8"), path.name


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
