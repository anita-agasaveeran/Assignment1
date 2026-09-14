.PHONY: help setup run run-fast serve test properties profile clean lint-palette

# Use a local venv if one exists, otherwise the shared one in the parent folder.
VENV := $(shell [ -x .venv/bin/python ] && echo .venv || echo ../.venv)
PY   := $(VENV)/bin/python
PORT ?= 8931

help:
	@echo "make setup       create the venv and install dependencies"
	@echo "make run         full CRISP-DM pipeline incl. AutoResearch (~2-3 min)"
	@echo "make run-fast    pipeline without hill climbing or the null baseline"
	@echo "make serve       dashboard + API on http://127.0.0.1:$(PORT)"
	@echo "make test        run the test suite"
	@echo "make properties  print the derived interestingness property matrix"
	@echo "make profile     data-understanding report only"

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

run:
	PYTHONPATH=src $(PY) -m armlab.cli run

run-fast:
	PYTHONPATH=src $(PY) -m armlab.cli run --no-autoresearch --no-null-baseline

serve:
	PYTHONPATH=src $(PY) -m armlab.cli serve --port $(PORT)

test:
	PYTHONPATH=src $(PY) -m pytest tests -q

properties:
	PYTHONPATH=src $(PY) -m armlab.cli properties

profile:
	PYTHONPATH=src $(PY) -m armlab.cli profile

lint-palette:
	$(PY) tools/validate_palette.py "#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4" --mode light
	$(PY) tools/validate_palette.py "#3987e5,#d95926,#199e70,#c98500,#d55181" --mode dark

clean:
	rm -rf artifacts/*.json artifacts/*.jsonl data/interim/*.parquet
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
