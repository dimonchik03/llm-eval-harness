# llm-eval-harness

[![ci](https://github.com/dimonchik03/llm-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/dimonchik03/llm-eval-harness/actions/workflows/ci.yml)

An evaluation harness for structured LLM outputs from call-transcript analysis:
call type, lead quality, custom checklist fields, risks and a narrative
summary. It measures whether a prompt or model change made that analysis
better or worse — and by how much, with confidence intervals instead of bare
point estimates.

The repo contains the measurement machinery only. The model under test is
supplied by you as a single function.

```
raw calls ─▶ stratified_sample ─▶ anonymize ─▶ EvalRecord JSONL ─▶ human labels
                                                    │
             predict_fn (your model/pipeline) ─▶ evaluate (optionally concurrent)
                                                    │
                    ┌───────────────────────────────┴──────────────────────────┐
                    │ per-stage metrics: accuracy + 95% CI, per-class P/R/F1,  │
                    │ confusion matrices, per-slug field scores — sliced       │
                    │ overall and per company / template / industry           │
                    │ + optional fail-closed LLM-as-judge faithfulness         │
                    └───────────────────────────────┬──────────────────────────┘
                                                    │
        JSON report + per-record rows + baseline deltas + regression gate (exit 1)
```

## Quickstart — no API keys needed

```bash
git clone https://github.com/dimonchik03/llm-eval-harness && cd llm-eval-harness
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make demo
```

`make demo` generates a fully synthetic dataset (invented companies, invented
people, Ukrainian/English call transcripts run through the real anonymizer) and
evaluates it with a deterministic offline predictor. No network, no keys.
The tail of the output:

```
[eval] dataset=examples/synthetic_demo.jsonl split=None records=22 evaluated=22 errors=0
[eval] overall      n=22   call_type 0.909 [0.722, 0.975]  lead_quality 0.875 [0.640, 0.965]  fields_f1 0.864  risks_f1 0.909
[eval] company 1    n=7    call_type 0.857 [0.487, 0.974]  lead_quality 1.000 [0.566, 1.000]  fields_f1 0.857  risks_f1 1.000
[eval] company 2    n=8    call_type 0.875 [0.529, 0.978]  lead_quality 0.833 [0.436, 0.970]  fields_f1 1.000  risks_f1 0.857
[eval] company 3    n=7    call_type 1.000 [0.646, 1.000]  lead_quality 0.800 [0.376, 0.964]  fields_f1 0.714  risks_f1 0.889
```

The demo predictor replays ground-truth labels with hash-seeded noise, so its
scores are meaningless by construction — it exists to exercise the full harness
path end to end (`examples/fake_predictor.py` says this loudly too). The
intervals are the useful part of the picture: with seven calls per company, a
perfect 1.000 still only means "somewhere above ~0.65".

## What's in the box

| Module | Purpose |
| --- | --- |
| `eval_harness/schema.py` | Typed `EvalRecord` = anonymized input + human label + stratum + split ([format docs](docs/dataset-format.md)) |
| `eval_harness/sampling.py` | Stratified sampling with per-company/per-template minimums; stratified train/holdout split |
| `eval_harness/anonymize.py` | PII stripping: email/URL/secret/phone maskers + known-name redaction that handles Ukrainian case endings |
| `eval_harness/dataset_io.py` | Atomic JSONL read/write |
| `eval_harness/metrics.py` | Pure-Python accuracy with Wilson CIs, per-class P/R/F1, confusion matrices, multi-label and per-slug scores |
| `eval_harness/runner.py` | `run_eval` / `evaluate` → JSON report and per-record rows; fail-closed LLM-as-judge; baseline deltas |
| `eval_harness/cli.py` | `llm-eval` command: splits, limits, concurrency, judge, report + rows output, baseline update and gate |
| `.github/workflows/ci.yml` | Lint, strict type check, tests on Python 3.11–3.13, reproducible-demo check |
| `.github/workflows/eval.yml` | A dormant-safe example of a real eval run in CI (see below) |

## Plugging in your model

The seam is one function: `predict_fn(record: EvalRecord) -> StagePrediction`.
Inside it you call your pipeline — assemble the prompt, hit your provider, map
the parsed output onto the labelled surface:

```python
from eval_harness import StagePrediction, estimate_cost_usd, read_records, run_eval


def predict(record):
    result = my_pipeline.analyze(record.input.transcript)  # your code
    return StagePrediction(
        call_id=record.input.call_id,
        company_id=record.input.company_id,
        template_id=record.input.template_id,
        call_type_key=result.call_type,
        is_analytical=result.is_analytical,
        lead_quality=result.lead_quality,
        custom_fields=result.fields,  # slug -> value
        risk_category_keys=result.risk_keys,
        summary=result.narrative,  # judged for faithfulness if --judge
        model=result.model,
        latency_ms=result.latency_ms,
        cost_usd=estimate_cost_usd(result.model, result.usage),
    )


report = run_eval(read_records("data/dataset_v1.jsonl"), predict, split="holdout", concurrency=8)
```

Or from the command line, naming the function as a `module:function` import path:

```bash
llm-eval --dataset data/dataset_v1.jsonl --split holdout \
    --predictor my_pipeline.eval_adapter:predict --concurrency 8 \
    --out report.json --predictions-out predictions.jsonl
```

LLM calls are I/O-bound, so `--concurrency` runs records on a thread pool;
results come back in input order and the report is identical to a sequential
run. Your `predict_fn` must then be thread-safe.

## Failure handling

Nothing fails silently, and nothing fails *optimistically*:

- **A raising `predict_fn` is recorded as an `error`** on that record — one bad
  call never aborts a run. Errored records are **excluded from the quality
  metrics** and reported as `n_errors` / `error_rate` per slice: a provider
  outage shows up as an outage, not as a drop in accuracy.
- **The judge fails closed.** A refusal, an unparseable answer or a score
  outside 1–5 becomes a `judge_error` on the record and is counted in
  `n_judge_errors` — never as a faithful summary.
- **A missing dataset is an error** (exit code 2), so a typo in a path cannot
  produce a green run.

## Reading the results

- **Report (`--out`)** — overall and per company / template / industry blocks:
  accuracy with a 95% Wilson interval, per-class P/R/F1 and a confusion matrix
  for call type, multi-label P/R/F1 for risks, per-slug P/R/F1 for custom
  fields and template extensions, latency p50/p95, cost, and a `run` block
  recording the predictor, judge model and settings.
- **Per-record rows (`--predictions-out`)** — one JSONL line per call with
  truth vs prediction, missing and extra risks, mismatched fields, judge verdict
  and an `all_correct` flag. The report says *how much* changed; the rows say
  *which calls*:

  ```bash
  jq -c 'select(.all_correct == false) | {call_id, call_type, risks}' predictions.jsonl
  ```

## LLM-as-judge (optional)

Classification-style fields are scored against human labels directly. The
narrative summary is the one field with no closed-form ground truth, so it gets
an LLM judge: a fixed evaluator model receives the transcript and the generated
narrative, and returns a structured verdict — `faithful: bool`, a 1–5
groundedness score, and the list of unsupported claims. Pin the judge model per
benchmark (`--judge-model`) so scores stay comparable across the models under
test.

```bash
pip install -e ".[judge]"     # adds the openai client
export OPENAI_API_KEY=...     # key for the judge model
llm-eval --dataset ... --predictor ... --judge --judge-model gpt-4o-mini
```

Tests monkeypatch the judge's single LLM seam, so the suite runs keyless.

## Baselines and the regression gate

`--update-baseline` snapshots a run's report into `eval_baseline.json` (or
`--baseline PATH`); later runs print headline deltas — call-type accuracy,
lead-quality accuracy, custom-fields micro-F1, risks micro-F1 — against it.
`--fail-on-regression 0.02` turns that into a gate: exit code 1 when any
headline metric drops by more than two points. Set the threshold with the
confidence intervals in mind; on a 50-record subset, a two-point move is noise.

## Anonymization

`anonymize_text(text, names=[...], company_names=[...])` runs at dataset-build
time, before a transcript is written anywhere:

- emails, URLs (`https://`, `www.`, `gs://`, `s3://`) and API-key-shaped strings
  become `[EMAIL]`, `[URL]`, `[SECRET]`;
- phone numbers become `[PHONE]` in the forms speech-to-text produces
  (`+380 67 123 45 67`, `067-123-45-67`, `(415) 555-0134`), while dates and
  space-grouped amounts next to a currency (`1 500 000 гривень`) are kept;
- the participants and companies you pass become `[PERSON_N]` / `[COMPANY]`,
  including Ukrainian case forms (`Олена` → `Олени`, `Оленою`; `Петро` →
  `Петра`) and company names without their legal form (`Globex GmbH` →
  `Globex`). All names are replaced in one pass, so a placeholder is never
  re-matched, and anonymizing twice changes nothing.

Known limits, by design: names that are not passed in are not detected (no
open-vocabulary NER), numbers spoken as words are not masked, and names whose
stem changes (`Лев` → `Льва`) are missed. Spot-check a sample of every new
dataset before it is shared.

## CI

`ci.yml` runs on every push and pull request: `ruff` lint and format check,
strict `mypy`, the test suite on Python 3.11–3.13, a check that the synthetic
dataset regenerates byte-identically, and the offline demo.

`eval.yml` is a **dormant-safe example** of a real eval run, not an enforced
gate: real datasets are git-ignored (they contain transcript text even after
anonymization), so its guard step skips the job cleanly until a fork commits a
dataset and provisions an `OPENAI_API_KEY` secret plus an `EVAL_PREDICTOR`
variable. Wired up, it runs a 50-record holdout subset with the judge, posts the
headline metrics as a PR comment and fails the check on a regression.

## Design notes

- **Stratified sampling, not random.** Call analysis is usually multi-tenant:
  companies differ in template, industry and call mix, and a random sample lets
  one large tenant dominate every metric. Sampling per
  `(company, template, call_type, duration_bucket)` stratum with per-company and
  per-template minimums keeps small tenants visible, and the report buckets by
  the same axes — a regression confined to one tenant shows up instead of
  averaging away.
- **Human labels first, LLM-as-judge only where labels can't reach.** Everything
  with a closed answer set (call type, lead quality, slug-keyed fields, risks)
  is scored against human ground truth with plain-Python metrics — cheap,
  deterministic, explainable. The judge is reserved for the one open-ended
  field (summary faithfulness), where "is every claim supported by the
  transcript?" is a task LLMs verify far more reliably than they generate.
- **Intervals, not just point estimates.** Per-tenant slices are small. Every
  accuracy carries a Wilson interval, which stays sensible at 0% and 100% and
  on a handful of records, so a "regression" inside the noise is visible as
  such.
- **Train/holdout split by stratum.** Prompt tuning iterates against the train
  slice; the holdout answers "did we overfit our prompt to the examples we
  stared at?". Splitting within each stratum keeps both sides representative of
  every tenant.
- **Anonymize at dataset-build time.** Datasets are meant to be built from real
  calls, so PII stripping happens before anything is written, not at display
  time. Known-name redaction was chosen over open-vocabulary NER deliberately —
  NER is unreliable on mixed Ukrainian/russian speech, while the system that
  recorded the call already knows who was on it.
- **The predict seam is a function, not a config.** Everything worth evaluating
  (a prompt variant, a provider swap, a whole pipeline refactor) is expressible
  as `predict_fn`, and everything the harness does — filtering, judging, error
  capture, bucketing, deltas — is independent of it. It also makes the harness
  trivially testable: the test suite runs the full path with an in-memory stub.
- **No third-party metrics deps.** The metrics are ~200 lines of plain Python.
  For this shape of task that is cheaper than dragging in a framework, and every
  number in a report can be traced to a hand-checkable function.

## Development

```bash
make check      # ruff lint + format check, strict mypy, pytest
make format     # ruff fix + format
```

## License

MIT — see [LICENSE](LICENSE).
