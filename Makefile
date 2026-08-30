PYTHON ?= python3

.PHONY: demo test lint typecheck format check

## Generate the synthetic dataset and run the harness over it — no network, no API keys.
demo:
	$(PYTHON) examples/generate_synthetic.py --out examples/synthetic_demo.jsonl
	$(PYTHON) -m eval_harness --dataset examples/synthetic_demo.jsonl --predictor examples.fake_predictor:predict

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

typecheck:
	$(PYTHON) -m mypy

format:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

## Everything CI runs.
check: lint typecheck test
