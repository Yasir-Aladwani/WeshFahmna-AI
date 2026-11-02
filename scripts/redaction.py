"""Publication-safe masking of personal identifiers in free-text fields.

Applied by scripts/prepare_data.py to every free-text column it writes into app_data/.
Only text is changed: identifiers are replaced with bracketed placeholders such as
[EMAIL] or [PHONE]. Numeric results, labels, timestamps, and row counts are untouched.

The rules are ordered: broader identifiers (emails, phone numbers) run before the
generic number rules so a phone number is not half-masked as [NUMBER].
"""
from __future__ import annotations

import re
from collections import Counter

_DIGIT = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double\s+\w+|triple\s+\w+)"
_DIGIT_OR_TO = rf"(?:{_DIGIT}|to)"
_EMAIL_STOP = (r"is|it's|its|as|email|e-mail|my|listed|so|then|and|okay|ok|yes|yeah|the|id|mail|right|me|"
               r"that|to|was|have|it|of|address|sent|send|provided|from|via|perfect|confirm")
_LOCAL_TOKEN = rf"(?<![\w.'-])(?!(?:{_EMAIL_STOP})\b)[a-z0-9][\w.'-]*"
_DOMAIN = r"(?:[a-z0-9-]+\s*(?:\.|\bdot\b)\s*)+(?:com|co\.uk|co|uk|net|org|in)\b"
_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"

# Customer, agent, and third-party surnames found in the 12 transcripts. First names are kept.
_SURNAMES = r"Smith|Johnson|Roscoe|Sankar|Mishra|McPherson|Manon|Menon|Nier"

RULES: list[tuple[str, re.Pattern, str]] = [
    # A spelled-out email that trails off before the domain.
    ("email", re.compile(r"\be r y k a h dot b l a c k eight four at g\b", re.I), "[EMAIL]"),
    # Letter-by-letter email addresses: "rocks-R-O-C-K-S-at-gmail.com", "E -R -Y ... -at -gmail.com".
    ("email", re.compile(rf"(?:\b[a-z]+\s?-\s?)?(?:\b[a-z]\s?-\s?){{2,}}at\s?-\s?{_DOMAIN}", re.I), "[EMAIL]"),
    # Written email addresses, including ASR spacing like "honey@ gmail.com".
    ("email", re.compile(r"[\w.+-]+\s?@\s?[\w-]+(?:\s?\.\s?[\w-]+)+", re.I), "[EMAIL]"),
    # Spoken email addresses: "<up to 6 tokens> at [the rate of|the red] gmail.com", "rocks-at-gmail.com".
    ("email", re.compile(rf"(?:{_LOCAL_TOKEN}[\s,]+){{0,6}}{_LOCAL_TOKEN}[\s-]*\bat\b(?:\s+the\s+(?:rate\s+of|red))?"
                         rf"[\s-]*{_DOMAIN}", re.I), "[EMAIL]"),
    ("phone", re.compile(r"\b\d{3}\s?-\s?\d{3}\s?-\s?\d{4}\b"), "[PHONE]"),
    ("phone", re.compile(r"\b(?:\d\s){6,}\d\b"), "[PHONE]"),
    ("phone", re.compile(r"\b0\d{8,9}\b|\b\d{10}\b"), "[PHONE]"),
    ("date_of_birth", re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?(?:\s+of)?\s+{_MONTH},?\s+(?:19|20)\d{{2}}\b"),
     "[DATE OF BIRTH]"),
    ("reference_number", re.compile(r"\b[A-Z]{0,3}\d{7,}\b"), "[REFERENCE]"),
    # Spoken digit strings (phone, card, loyalty, booking, purchase numbers): 4+ digit words in a row.
    ("spoken_number", re.compile(rf"\b{_DIGIT_OR_TO}(?:[\s,.]+{_DIGIT_OR_TO}){{3,}}\b", re.I), "[NUMBER]"),
    # Bare identifiers quoted in summaries, e.g. "security card number (8783)", "code (456)".
    ("identifier", re.compile(r"\((\d{3,})\)"), "([NUMBER])"),
    # Last four digits of a payment card, and a spoken one-time code.
    ("card_digits", re.compile(r"(last four digits[^?]{0,80}\?\s*(?:yeah|yes)?,?\s*)\d{4}\b", re.I),
     r"\1[CARD DIGITS]"),
    ("one_time_code", re.compile(r"(read the code for me\?\s*(?:yeah|yes)?,?\s*(?:it's|it is)?\s*)\d{3,6}\b", re.I),
     r"\1[CODE]"),
    # Letter-by-letter name spellings: "E-R-Y-K-A-H", "e r y k a h".
    ("spelled_name", re.compile(r"\b(?:[A-Za-z]\s?-\s?){2,}[A-Za-z]\b"), "[SPELLED NAME]"),
    ("spelled_name", re.compile(r"\b(?:[a-z] ){3,}[a-z]\b"), "[SPELLED NAME]"),
    # Remainder of a spelled email split across turns: "[SPELLED NAME] dot [SPELLED NAME] eight four at g".
    ("email", re.compile(r"\[SPELLED NAME\](?:\s+dot\s+\[SPELLED NAME\])*(?:\s+[a-z]+){0,3}\s+at\s+g\b", re.I),
     "[EMAIL]"),
    ("surname", re.compile(rf"\b(?:{_SURNAMES})\b"), "[SURNAME]"),
    # "Black" only as a surname, never the colour ("black and gold heels").
    ("surname", re.compile(r"\b((?:Miss|Ms|Mrs|Mr)\.?\s+|(?:Erika|Erica|Erykah)\s+)Black\b", re.I), r"\1[SURNAME]"),
]


# Context-aware masking of dictated numbers (phone, card, code, booking digits) that the
# pattern rules cannot recognise on their own, e.g. "Eight six. Yeah. Five four three."
PROMPT = re.compile(r"(?:telephone|phone|mobile|contact)\s+number|last\s+(?:four|three|4|3)\s+digits|"
                    r"read (?:the|that) code|security card|loyalty card|confirmation number|purchase number|"
                    r"give me that number|read that back|please write", re.I)
_NUM_WORD = r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|double|triple)"
_UNIT_WORDS = (r"%|percent|dollars?|pounds?|rupees?|per\b|days?|gb\b|plans?\b|plants?\b|a\.?m\b|p\.?m\b|"
               r"minutes?|hours?|seconds?|limit|sms|more\b|or\b|of\b")
_RUN = re.compile(rf"(?<![$£₹\w])(?:{_NUM_WORD}|\d+)(?:[\s,.-]+(?:{_NUM_WORD}|\d+))*\b(?!\s*(?:{_UNIT_WORDS}))",
                  re.I)
_WINDOW = 120


def _is_identifier_run(run: str, min_words: int) -> bool:
    return len(re.findall(r"\w+", run)) >= min_words or bool(re.fullmatch(r"\d{3,}", run.strip()))


def _mask_runs(text: str, counts: Counter | None, min_words: int, spans=None) -> tuple[str, int]:
    """Mask number runs (optionally only those starting inside ``spans``); return text and hit count."""
    hits = 0

    def repl(m):
        nonlocal hits
        inside = spans is None or any(a <= m.start() < b for a, b in spans)
        if inside and _is_identifier_run(m.group(0), min_words):
            hits += 1
            return "[NUMBER]"
        return m.group(0)

    text = _RUN.sub(repl, text)
    if counts is not None and hits:
        counts["dictated_number"] += hits
    return text, hits


def _digit_dominant(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    nums = [t for t in tokens if re.fullmatch(rf"{_NUM_WORD}|\d+", t)]
    real = [t for t in nums if t != "oh"]  # "Oh," alone is an interjection, not a digit
    return bool(real) and len(nums) / len(tokens) >= 0.5 and not re.search(rf"{_UNIT_WORDS}|[$£₹]", text, re.I)


def redact(text: str, counts: Counter | None = None) -> str:
    """Return ``text`` with personal identifiers masked; tally rule hits into ``counts``."""
    if not isinstance(text, str) or not text:
        return text
    for category, pattern, replacement in RULES:
        text, n = pattern.subn(replacement, text)
        if counts is not None and n:
            counts[category] += n
    spans = [(m.end(), m.end() + _WINDOW) for m in PROMPT.finditer(text)]
    if spans:
        text, _ = _mask_runs(text, counts, min_words=2, spans=spans)
    return text


def redact_turns(turns, counts: Counter) -> list:
    """Redact time-ordered transcript turns, carrying number-dictation context across turns.

    ``turns`` must be sorted by call_id and start. A turn is treated as dictation when it is
    mostly digits, or when the previous turn in the same call asked for or dictated a number.
    """
    out, prev_call, prev_prompt = [], None, False
    for call_id, text in zip(turns["call_id"], turns["text"]):
        if call_id != prev_call:
            prev_prompt = False
        red, hits = redact(text, counts), 0
        if isinstance(red, str):
            if _digit_dominant(red):
                red, hits = _mask_runs(red, counts, min_words=1)
            elif prev_prompt:
                red, hits = _mask_runs(red, counts, min_words=2)
        out.append(red)
        prev_prompt = bool(hits) or (isinstance(text, str) and bool(PROMPT.search(text)))
        prev_call = call_id
    return out


def redact_columns(df, columns, counts: Counter):
    """Redact the given free-text columns of a DataFrame copy."""
    out = df.copy()
    for col in columns:
        if col in out:
            out[col] = out[col].map(lambda v: redact(v, counts))
    return out


# Patterns that must not survive redaction; used by prepare_data.py and the tests.
RESIDUAL_CHECKS = {
    "email": re.compile(r"@|\bat\s+(?:the\s+(?:rate\s+of|red)\s+)?(?:gmail|yahoo|hotmail|outlook)\b|"
                        r"\b[\w-]+\s*(?:\.|\bdot\b)\s*(?:com|co\.uk|co)\b|\bjavu\b", re.I),
    "phone": re.compile(r"\b\d{3}\s?-\s?\d{3}\s?-\s?\d{4}\b|\b(?:\d\s?){9,}\b"),
    "surname": re.compile(rf"\b(?:{_SURNAMES})\b|\b(?:Miss|Erika|Erica)\s+Black\b", re.I),
}
