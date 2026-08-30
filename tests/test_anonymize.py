"""Anonymizer tests — the PII guarantees the dataset relies on.

Phone formats are the ones speech-to-text actually produces; names cover the
Ukrainian case endings that change the last letter of a name.
"""

import pytest

from eval_harness.anonymize import anonymize_text, mask_patterns, redact_names


# --------------------------------------------------------------------------- #
# Pattern maskers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "+380 67 123 45 67",
        "+38 (067) 123-45-67",
        "067-123-45-67",
        "0671234567",
        "(415) 555-0134",
        "+1 415 555 0134",
        "+420 601 123 456",
        "067.123.45.67",
    ],
)
def test_phone_formats_are_masked(text):
    assert mask_patterns(f"мій номер {text}, дзвоніть") == "мій номер [PHONE], дзвоніть"


@pytest.mark.parametrize(
    "text",
    [
        "15.01.2026",
        "2026-01-15",
        "о 10:30",
        "до 60000 гривень",
        "1 500 000 гривень",
        "$2 000 000",
        "тариф на 1500 дзвінків",
    ],
)
def test_dates_amounts_and_short_numbers_survive(text):
    assert mask_patterns(text) == text


def test_long_account_numbers_are_masked_as_number():
    assert mask_patterns("IBAN DE89 3704 0044 0532 0130 00") == "IBAN DE89 [NUMBER]"


def test_email_url_and_secret_are_masked():
    text = (
        "пишіть на sales@globex.de, див. https://globex.de/pricing?ref=abc і www.globex.de/x; key sk-abcdefghijklmnop"
    )
    assert mask_patterns(text) == "пишіть на [EMAIL], див. [URL] і [URL]; key [SECRET]"


def test_bucket_uris_are_masked():
    assert mask_patterns("file at gs://bucket/calls/123.wav") == "file at [URL]"


# --------------------------------------------------------------------------- #
# Known-name redaction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "declined"),
    [
        ("Олена Ковальчук", "Олени"),
        ("Олена Ковальчук", "Оленою"),
        ("Олена Ковальчук", "Олено"),
        ("Петро Іваненко", "Петра"),
        ("Андрій Бондаренко", "Андрієм"),
        ("Іван Петренко", "Іване"),
        ("Олена Ковальчук", "Ковальчука"),
        ("Ірина Шевченко", "Шевченка"),
    ],
)
def test_declined_ukrainian_names_are_redacted(name, declined):
    assert redact_names(f"Дякую, {declined}, до зв'язку", [name]) == "Дякую, [PERSON_1], до зв'язку"


def test_full_name_becomes_one_placeholder():
    assert redact_names("Олену Ковальчук чекаємо", ["Олена Ковальчук"]) == "[PERSON_1] чекаємо"


def test_people_keep_stable_distinct_numbers():
    text = "John called Mary. Then Mary called John."
    expected = "[PERSON_1] called [PERSON_2]. Then [PERSON_2] called [PERSON_1]."
    assert redact_names(text, ["John Smith", "Mary Jones"]) == expected


def test_placeholder_is_never_rematched_by_a_later_name():
    # "Per" is a prefix of "PERSON"; a sequential replace would turn [PERSON_1] into [[PERSON_2]].
    assert redact_names("Per and John talked", ["John Smith", "Per"]) == "[PERSON_2] and [PERSON_1] talked"


def test_company_whole_word_and_legal_form():
    text = "Globex GmbH, просто Globex, але не Globexy; Auto vs Automobile"
    out = anonymize_text(text, company_names=["Globex GmbH", "Auto"])
    assert out == "[COMPANY], просто [COMPANY], але не Globexy; [COMPANY] vs Automobile"


def test_short_company_names_are_skipped():
    assert anonymize_text("AB testing", company_names=["AB"]) == "AB testing"


def test_anonymize_text_end_to_end_and_idempotent():
    text = "Олена Ковальчук з Acme Corp: пишіть на olena.kovalchuk@example.com або +380 97 123 45 67. Дякую, Олено!"
    kwargs = {"names": ["Олена Ковальчук"], "company_names": ["Acme Corp"]}
    out = anonymize_text(text, **kwargs)
    assert out == "[PERSON_1] з [COMPANY]: пишіть на [EMAIL] або [PHONE]. Дякую, [PERSON_1]!"
    assert anonymize_text(out, **kwargs) == out


def test_empty_input():
    assert anonymize_text(None) == ""
    assert anonymize_text("") == ""
