"""Anonymization for eval datasets — strip PII before a transcript enters a dataset.

Two layers, applied in this order:

1. **Pattern maskers** replace structured PII with typed placeholders: emails ->
   ``[EMAIL]``, URLs (incl. ``gs://`` / ``s3://``) -> ``[URL]``, API-key-shaped
   strings -> ``[SECRET]``, phone numbers -> ``[PHONE]`` and longer digit
   sequences (account numbers) -> ``[NUMBER]``. Phone numbers are matched in the
   forms speech-to-text actually emits — ``+380 67 123 45 67``,
   ``067-123-45-67``, ``(415) 555-0134`` — not only as unbroken digit runs.
   Date-shaped strings (``2026-01-15``, ``15.01.2026``) and space-grouped amounts
   next to a currency (``1 500 000 гривень``) are left alone.
2. **Known-name redaction** replaces the names the caller already knows (speaker
   labels, the CRM contact, the manager, the company) with stable
   ``[PERSON_N]`` / ``[COMPANY]`` placeholders.

There is deliberately no open-vocabulary NER: it is unreliable on mixed
Ukrainian/russian speech, while the caller usually knows exactly who was on the
call.

Ukrainian and Russian decline names, and the ending changes with the case:
``Олена`` -> ``Олени`` / ``Оленою``, ``Петро`` -> ``Петра``, ``Андрій`` ->
``Андрія``. For Cyrillic name tokens the matcher drops one declinable ending
(a final vowel, soft sign or ``й``) and matches the stem plus any ending. It
accepts rare over-redaction of an unrelated word that shares the stem
(``Марина`` also blanks ``маринад``) as the safe error for PII. Names whose stem
itself changes (``Лев`` -> ``Льва``) are not covered.

Company names match as whole words only (a company called ``Auto`` does not
blank ``Automobile``); a trailing or leading legal form is also stripped, so
``Globex GmbH`` catches a bare ``Globex`` too. Company strings shorter than
three characters are skipped.

All names are replaced in a single regex pass, and existing placeholders are
matched first and kept as they are, so a name can never re-match a placeholder
inserted for another name, and anonymizing twice changes nothing.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

# Name tokens shorter than this are not redacted on their own (a lone two-letter
# token collides with too many common words). Also the minimum company length.
_MIN_TOKEN_LEN = 3

# --------------------------------------------------------------------------- #
# Pattern maskers
# --------------------------------------------------------------------------- #
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL_RE = re.compile(r"(?:\b(?:https?|gs|s3)://|\bwww\.)[^\s\"'<>]*[^\s\"'<>.,;:!?)\]]", re.IGNORECASE)
_SECRET_RE = re.compile(r"\b(?:sk|rk|pk|api|key|token)[-_][A-Za-z0-9_-]{10,}\b", re.IGNORECASE)

# A number candidate: optional "+" / "(", then digits joined by the separators
# speech-to-text emits (space, no-break space, dash, dot, parentheses). Whether a
# candidate is a phone, a date or an amount is decided in code.
_NUMBER_RE = re.compile(r"(?<![\w+])\+?\(?\d[\d \u00a0().-]{4,}\d(?!\w)")
_DATE_RE = re.compile(r"\d{1,4}([./-])\d{1,2}\1\d{1,4}")
_GROUPED_AMOUNT_RE = re.compile(r"\d{1,3}(?:[ \u00a0]\d{3})+")
_CURRENCY_AFTER_RE = re.compile(
    r"[ \u00a0]*(?:грн|гривень|гривні|гривня|uah|usd|eur|€|\$|доларів|долари|dollars?|euros?|євро)(?!\w)",
    re.IGNORECASE,
)
_CURRENCY_BEFORE_RE = re.compile(r"(?:\$|€|usd|eur|uah)[ \u00a0]*$", re.IGNORECASE)
_MIN_PHONE_DIGITS = 7
_MAX_PHONE_DIGITS = 15  # E.164 upper bound


def _next_to_currency(match: re.Match[str]) -> bool:
    text = match.string
    if _CURRENCY_AFTER_RE.match(text, match.end()):
        return True
    return bool(_CURRENCY_BEFORE_RE.search(text[max(0, match.start() - 8) : match.start()]))


def _mask_number(match: re.Match[str]) -> str:
    candidate = match.group(0)
    digits = sum(ch.isdigit() for ch in candidate)
    if digits < _MIN_PHONE_DIGITS or _DATE_RE.fullmatch(candidate):
        return candidate
    if _GROUPED_AMOUNT_RE.fullmatch(candidate) and _next_to_currency(match):
        return candidate
    return "[PHONE]" if digits <= _MAX_PHONE_DIGITS else "[NUMBER]"


def mask_patterns(text: str) -> str:
    """Replace emails, URLs, secret-shaped tokens and phone/account numbers with placeholders."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _URL_RE.sub("[URL]", text)
    text = _SECRET_RE.sub("[SECRET]", text)
    return _NUMBER_RE.sub(_mask_number, text)


# --------------------------------------------------------------------------- #
# Known-name redaction
# --------------------------------------------------------------------------- #
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_DECLINABLE_ENDINGS = frozenset("аяоеєиіїуюьй")
_LEGAL_FORMS = frozenset(
    {"gmbh", "ag", "se", "kg", "ug", "inc", "llc", "ltd", "corp", "co", "plc", "sa", "bv", "тов", "пп", "пат", "прат"}
)
# Placeholders this module emits; matched first and kept verbatim.
_PLACEHOLDER = r"(?-i:\[(?:PERSON_\d+|COMPANY|EMAIL|URL|SECRET|PHONE|NUMBER)\])"


def _stem(token: str) -> str:
    """Drop one declinable ending from a Cyrillic name token: ``Олена`` -> ``Олен``."""
    if _CYRILLIC_RE.search(token) and len(token) > _MIN_TOKEN_LEN and token[-1].casefold() in _DECLINABLE_ENDINGS:
        return token[:-1]
    return token


def _person_fragments(name: str) -> list[str]:
    """Regex fragments for one person: the full name, then each long-enough token."""
    tokens = name.split()
    fragments = [r"\s+".join(re.escape(_stem(token)) + r"\w*" for token in tokens)]
    if len(tokens) > 1:
        fragments.extend(re.escape(_stem(token)) + r"\w*" for token in tokens if len(token) >= _MIN_TOKEN_LEN)
    return fragments


def _company_fragments(company: str) -> list[str]:
    """Whole-word regex fragments for a company name, with and without its legal form."""
    tokens = company.replace(",", " ").split()
    core = [token for token in tokens if token.strip(".").casefold() not in _LEGAL_FORMS]
    variants = {" ".join(tokens), " ".join(core)}
    return [
        r"\s+".join(re.escape(token) for token in variant.split())
        for variant in variants
        if len(variant) >= _MIN_TOKEN_LEN
    ]


def _redact(text: str, names: Iterable[str], company_names: Iterable[str]) -> str:
    people = [name.strip() for name in names if name and name.strip()]
    companies = [company.strip() for company in company_names if company and company.strip()]

    # (fragment, group name); longest fragment first so the most specific variant wins.
    alternatives: list[tuple[str, str]] = []
    for index, name in enumerate(people, start=1):
        alternatives.extend((fragment, f"p{index}_{k}") for k, fragment in enumerate(_person_fragments(name)))
    company_alternatives = sorted({f for c in companies for f in _company_fragments(c)}, key=len, reverse=True)
    if not alternatives and not company_alternatives:
        return text
    alternatives.sort(key=lambda item: len(item[0]), reverse=True)

    branches = [f"(?P<keep>{_PLACEHOLDER})"]
    if company_alternatives:
        branches.append(rf"(?P<company>(?<!\w)(?:{'|'.join(company_alternatives)})(?!\w))")
    branches.extend(rf"(?P<{group}>(?<!\w){fragment})" for fragment, group in alternatives)
    pattern = re.compile("|".join(branches), re.IGNORECASE)

    def replace(match: re.Match[str]) -> str:
        group = match.lastgroup or "keep"
        if group == "keep":
            return match.group(0)
        if group == "company":
            return "[COMPANY]"
        return f"[PERSON_{group[1:].split('_')[0]}]"

    return pattern.sub(replace, text)


def redact_names(text: str, names: Iterable[str]) -> str:
    """Replace each known personal name (and its declined forms) with a stable ``[PERSON_k]``."""
    return _redact(text, names, ())


def anonymize_text(
    text: str | None,
    *,
    names: Iterable[str] = (),
    company_names: Iterable[str] = (),
) -> str:
    """Mask structured PII, then redact the known people and companies."""
    if not text:
        return ""
    return _redact(mask_patterns(text), names, company_names)
