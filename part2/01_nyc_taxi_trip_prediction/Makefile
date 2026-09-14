.PHONY: all data train eda serve test ablation clean help
PY := python

help:
	@echo "make data     download + clean the trip records (cached)"
	@echo "make train    fit the point model and both quantile models"
	@echo "make eda      dashboard stats + report figures"
	@echo "make all      data -> train -> eda"
	@echo "make serve    run the API and front end on :8000"
	@echo "make test     run the test suite"
	@echo "make ablation settle the collinear-importance question"

all: data train eda

data:
	PYTHONPATH=src $(PY) -m taxi.ingest

train:
	PYTHONPATH=src $(PY) -m taxi.train

eda:
	PYTHONPATH=src $(PY) -m taxi.eda

serve:
	$(PY) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

test:
	OMP_NUM_THREADS=1 $(PY) -m pytest tests/ -q

ablation:
	$(PY) scripts/ablation.py

clean:
	rm -rf data/interim/* data/processed/* models/*.joblib models/*.json
	find . -name __pycache__ -type d -exec rm -rf {} +
