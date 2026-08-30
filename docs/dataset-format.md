# Eval dataset format

Stratified evaluation datasets for measuring LLM-analysis quality per
company / template / call-type. **Real datasets are git-ignored** (`*.jsonl`) —
they hold anonymized but still-sensitive transcript text. The only tracked
dataset is the invented `examples/synthetic_demo.jsonl`.

## Format

One JSON object per line (`EvalRecord`, see `eval_harness/schema.py`):

```json
{
  "input":   { "call_id": 1, "company_id": 3, "template_id": 5, "duration_bucket": "long",
               "transcript": "manager: ...\nclient: ...", "speaker_turns": [ { "speaker": "manager", "text": "..." } ] },
  "label":   { "call_type_key": null, "lead_quality": null, "custom_fields": {},
               "extensions": {}, "risk_category_keys": [], "summary_quality": null,
               "industry": null, "labelled_at": null },
  "stratum": { "company_id": 3, "template_id": 5, "call_type_key": "inbound", "duration_bucket": "long" },
  "split":   "train"
}
```

- Custom fields and extensions are keyed by **slug**, never by position, so
  labels stay valid as templates evolve.
- `lead_quality`, `duration_bucket` and `split` are validated against fixed
  value sets (`LEAD_QUALITY_VALUES`, `DURATION_BUCKETS`, `train`/`holdout`).
  A known `duration_sec` always determines `duration_bucket`.
- `industry` is an optional human-labelled field — it is rarely available as
  structured data, so it is not a sampling axis.
- `labelled_at` must be an ISO-8601 timestamp.
- A record counts as labelled once `label.labelled_at` is set (a human saved
  it). `run_eval` skips unlabelled records by default — no metric value, no
  wasted LLM budget.
- `split` is `train` / `holdout` / `null`, assigned per stratum by
  `stratified_split` so the holdout mirrors the training stratification.

## Building a dataset

```bash
# Synthetic demo data (deterministic, safe to commit):
python examples/generate_synthetic.py --out examples/synthetic_demo.jsonl

# Your own data: sample, anonymize, write.
```

```python
from eval_harness import anonymize_text, stratified_sample, stratified_split, write_records

calls = load_your_calls()  # whatever your source of finished, transcribed calls is
sample = stratified_sample(
    calls,
    stratum_of=lambda c: (c.company_id, c.template_id, c.call_type_key, bucket(c)),
    company_of=lambda c: c.company_id,
    template_of=lambda c: c.template_id,
)
records = [to_eval_record(c) for c in sample]  # anonymize_text() every transcript here
train, holdout = stratified_split(records, stratum_of=lambda r: r.stratum.as_tuple())
```

Run every transcript through `anonymize_text(text, names=..., company_names=...)`
**before** it is written anywhere: emails, URLs, API-key-shaped strings and
phone numbers become `[EMAIL]` / `[URL]` / `[SECRET]` / `[PHONE]`, and the known
participant / company names you pass become stable `[PERSON_N]` / `[COMPANY]`
placeholders, Ukrainian case forms included. See the README's "Anonymization"
section for what it does not catch.

## Labelling

Labels are written by humans, not by the model under test. A small labelling
UI (Streamlit, for example) works well for that step; none ships with this
repo. Any editor that can update the `label` object of
a JSONL line works — set `labelled_at` (ISO-8601) when a label is final.
