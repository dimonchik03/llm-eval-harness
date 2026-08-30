#!/usr/bin/env python3
"""Generate a synthetic eval dataset for the demo.

Everything here is invented — companies, people, phone numbers, transcripts and
labels. No real call data is involved. The generator still runs the harness's
own anonymizer and splitter, so the output is structurally identical to a real
dataset:

- transcripts (uk/en, 5-10 turns) pass through :func:`eval_harness.anonymize_text`
  with the invented names, so the file shows exactly what anonymized records look
  like (``[PERSON_N]`` / ``[COMPANY]`` / ``[EMAIL]`` / ``[PHONE]`` placeholders);
- the train/holdout split comes from :func:`eval_harness.stratified_split`;
- records are written with :func:`eval_harness.write_records`.

Output is deterministic for a given ``--seed``, so the committed
``examples/synthetic_demo.jsonl`` regenerates byte-identically.
"""

from __future__ import annotations

import argparse
from typing import Any

from eval_harness import (
    CallInput,
    CallLabel,
    EvalRecord,
    SpeakerTurn,
    StratumKey,
    anonymize_text,
    duration_bucket,
    stratified_split,
    write_records,
)

DEFAULT_OUT = "examples/synthetic_demo.jsonl"
DEFAULT_SEED = 1337
LABELLED_AT = "2026-01-15T09:00:00+00:00"

# (company_id, company_name, template_id, template_name) — all invented.
COMPANIES: list[tuple[int, str, int, str]] = [
    (1, "Acme Corp", 1, "sales_calls_v2"),
    (2, "Globex GmbH", 1, "sales_calls_v2"),
    (3, "Initech Solutions", 2, "mixed_calls_v1"),
]

MANAGERS: list[str] = ["Emily Carter", "Ірина Шевченко", "Mark Evans"]

# (client_name, client_email, client_phone) — all invented.
CLIENTS: list[tuple[str, str, str]] = [
    ("John Miller", "john.miller@example.com", "+14155550134"),
    ("Олена Ковальчук", "olena.kovalchuk@example.com", "+380 97 123 45 67"),
    ("Peter Novak", "peter.novak@example.com", "+420 601 123 456"),
    ("Тарас Мельник", "taras.melnyk@example.com", "067-111-22-33"),
    ("Laura Bennett", "laura.bennett@example.com", "(323) 555-0188"),
    ("Андрій Бондаренко", "andrii.bondarenko@example.com", "+380509998877"),
]

# Each scenario: dialogue turns with placeholders + the ground-truth label a
# human would assign. ``label: None`` fields stay empty for non-analytical calls.
SCENARIOS: list[dict[str, Any]] = [
    {
        "key": "en_pricing",
        "call_type_key": "sales_consultation",
        "duration_sec": 310,
        "turns": [
            ("manager", "Good afternoon, {company}, {manager} speaking. How can I help you?"),
            (
                "client",
                "Hi, this is {client}. I saw your call-analytics platform online and I'd like to hear about pricing.",
            ),
            ("manager", "Happy to walk you through it. How many calls does your team handle per month?"),
            ("client", "Around two thousand, mostly outbound. Our budget is about 5000 dollars a year."),
            ("manager", "That fits our Growth plan — transcription plus AI analysis for up to three thousand calls."),
            ("client", "Sounds reasonable, though I still need to compare it with one more vendor before deciding."),
            ("manager", "Of course. I'll send a detailed quote today — what's the best email for you?"),
            ("client", "Use {client_email}. If anything urgent, my direct number is {client_phone}."),
            ("manager", "Perfect, expect the quote within the hour. Thank you, {client}!"),
        ],
        "label": {
            "lead_quality": "hot",
            "custom_fields": {"budget": "5000", "decision_maker": "yes"},
            "extensions": {"current_funnel_stage": "negotiation", "next_step": "send_quote"},
            "risk_category_keys": ["competitor_mention"],
        },
    },
    {
        "key": "en_demo_request",
        "call_type_key": "sales_consultation",
        "duration_sec": 240,
        "turns": [
            ("manager", "Hello, {company}, this is {manager}."),
            ("client", "Hi, {client} here. A colleague recommended your product — could we get a demo?"),
            ("manager", "Absolutely. What does your sales workflow look like today?"),
            ("client", "Mostly spreadsheets. We want to roll out something proper next month."),
            ("manager", "Next month is realistic. I can show you the analysis dashboard on Thursday."),
            ("client", "Thursday afternoon works. Send the invite to {client_email}."),
            ("manager", "Done. Looking forward to it, {client}!"),
        ],
        "label": {
            "lead_quality": "warm",
            "custom_fields": {"timeline": "next month"},
            "extensions": {"current_funnel_stage": "warm", "next_step": "schedule_demo"},
            "risk_category_keys": [],
        },
    },
    {
        "key": "uk_pricing",
        "call_type_key": "sales_consultation",
        "duration_sec": 280,
        "turns": [
            ("manager", "Доброго дня, компанія {company}, мене звати {manager}. Чим можу допомогти?"),
            ("client", "Вітаю, це {client}. Хочу дізнатися про тарифи на аналітику дзвінків."),
            ("manager", "Звісно. Скільки дзвінків на місяць обробляє ваш відділ продажів?"),
            ("client", "Приблизно тисяча. Бюджет — до 60000 гривень на рік."),
            ("manager", "Тоді вам підійде тариф «Стандарт», він покриває до півтори тисячі дзвінків."),
            ("client", "Дорогувато, чесно кажучи. У конкурентів бачив дешевше."),
            ("manager", "При річній оплаті є знижка. Надішлю детальний розрахунок сьогодні."),
            ("client", "Добре, чекаю. Пишіть на {client_email}."),
        ],
        "label": {
            "lead_quality": "warm",
            "custom_fields": {"budget": "60000"},
            "extensions": {"current_funnel_stage": "warm", "next_step": "send_quote"},
            "risk_category_keys": ["price_objection", "competitor_mention"],
        },
    },
    {
        "key": "uk_support",
        "call_type_key": "support_request",
        "duration_sec": 200,
        "turns": [
            ("manager", "Доброго дня, підтримка {company}, {manager} слухає."),
            ("client", "Це {client}, ми ваші клієнти вже рік. Після вчорашнього оновлення не працює експорт звітів."),
            ("manager", "Перепрошую за незручності. Яке повідомлення про помилку ви бачите?"),
            ("client", "Просто порожня сторінка. Нам ці звіти потрібні щопонеділка для керівництва."),
            ("manager", "Зрозуміло, передаю інженерам із високим пріоритетом. Оновлення буде сьогодні."),
            ("client", "Якщо до кінця тижня не виправите, будемо розглядати інші рішення."),
            ("manager", "Виправимо раніше. Напишу вам на {client_email}, щойно буде готово."),
        ],
        "label": {
            "lead_quality": "repeat_active",
            "custom_fields": {"issue_type": "export_failure"},
            "extensions": {"next_step": "escalate_to_engineering"},
            "risk_category_keys": ["churn_risk"],
        },
    },
    {
        "key": "en_wrong_number",
        "call_type_key": "wrong_number",
        "duration_sec": 30,
        "turns": [
            ("manager", "Good morning, {company}, {manager} speaking."),
            ("client", "Oh — sorry, I was trying to reach a pharmacy."),
            ("manager", "No worries, this is a software company."),
            ("client", "My mistake, wrong number. Have a good day."),
            ("manager", "You too, goodbye."),
        ],
        "label": None,
    },
    {
        "key": "uk_spam",
        "call_type_key": "spam",
        "duration_sec": 45,
        "turns": [
            ("manager", "Доброго дня, компанія {company}."),
            ("client", "Вітаю! Пропонуємо розміщення реклами у бізнес-каталозі, лише сьогодні знижка."),
            ("manager", "Дякую, нам це не цікаво."),
            ("client", "Можливо, вашому керівництву буде цікаво? Перший місяць безкоштовно."),
            ("manager", "Ні, дякую. Гарного дня."),
        ],
        "label": None,
    },
]


def _build_record(
    call_id: int,
    company: tuple[int, str, int, str],
    scenario: dict[str, Any],
    manager: str,
    client: tuple[str, str, str],
    *,
    labelled: bool,
) -> EvalRecord:
    company_id, company_name, template_id, template_name = company
    client_name, client_email, client_phone = client

    context = {
        "company": company_name,
        "manager": manager,
        "client": client_name,
        "client_email": client_email,
        "client_phone": client_phone,
    }
    raw_turns = [(speaker, text.format(**context)) for speaker, text in scenario["turns"]]
    anon = lambda text: anonymize_text(text, names=[manager, client_name], company_names=[company_name])  # noqa: E731
    transcript = anon("\n".join(f"{speaker}: {text}" for speaker, text in raw_turns))
    speaker_turns = [SpeakerTurn(speaker=speaker, text=anon(text)) for speaker, text in raw_turns]

    duration_sec = scenario["duration_sec"] + (call_id % 30) - 15
    call_input = CallInput(
        call_id=call_id,
        company_id=company_id,
        template_id=template_id,
        template_name=template_name,
        duration_sec=duration_sec,
        duration_bucket=duration_bucket(duration_sec),
        transcript=transcript,
        speaker_turns=speaker_turns,
    )
    stratum = StratumKey(
        company_id=company_id,
        template_id=template_id,
        call_type_key=scenario["call_type_key"],
        duration_bucket=call_input.duration_bucket,
    )

    label = CallLabel()
    if labelled:
        fields = dict(scenario["label"] or {})
        label = CallLabel(
            call_type_key=scenario["call_type_key"],
            labelled_by="synthetic",
            labelled_at=LABELLED_AT,
            **fields,
        )
    return EvalRecord(input=call_input, label=label, stratum=stratum)


def build_records(seed: int = DEFAULT_SEED) -> list[EvalRecord]:
    """Build the full synthetic dataset: 8 calls per company, split train/holdout."""
    # Two extra pricing calls per company so per-stratum groups are big enough to split.
    plan = [*SCENARIOS, SCENARIOS[0], SCENARIOS[2]]
    records: list[EvalRecord] = []
    call_id = 1000
    for index, company in enumerate(COMPANIES):
        for offset, scenario in enumerate(plan):
            call_id += 1
            manager = MANAGERS[(index + offset) % len(MANAGERS)]
            client = CLIENTS[call_id % len(CLIENTS)]
            # Leave the last call of two companies unlabelled to demo the labelled-only skip.
            labelled = not (offset == len(plan) - 1 and company[0] in (1, 3))
            records.append(_build_record(call_id, company, scenario, manager, client, labelled=labelled))

    train, holdout = stratified_split(records, stratum_of=lambda r: r.stratum.as_tuple(), seed=seed)
    for record in train:
        record.split = "train"
    for record in holdout:
        record.split = "holdout"
    return sorted(records, key=lambda r: r.input.call_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic demo eval dataset.")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Split seed (keeps output deterministic).")
    args = parser.parse_args(argv)

    records = build_records(seed=args.seed)
    count = write_records(args.out, records)
    labelled = sum(1 for record in records if record.label.is_labelled)
    print(f"[synthetic] wrote {count} records ({labelled} labelled) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
