PY := .venv/bin/python
PORT ?= 8099
CORPUS := data/processed/stories

.PHONY: help setup download data sft-data autoresearch champion pretrain sft evaluate dashboard chat test clean all

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## create venv and install dependencies
	python3 -m venv .venv && $(PY) -m pip install -U pip -r requirements.txt

download:  ## fetch the TinyStories corpora (byte-range slices, ~230MB total)
	bash scripts/download_data.sh

data:  ## CRISP-DM 2-3: profile, dedup, leakage audit, train tokenizer, tokenize
	$(PY) -m slm.data prepare-pretrain --input data/raw/stories_train.slice.txt \
	  --val data/raw/stories_valid.txt --out $(CORPUS) --vocab-size 8192

sft-data:  ## build the ChatML instruction dataset
	$(PY) -m slm.chat build-sft

autoresearch:  ## CRISP-DM 4-5: hill climb over the paper registry (~75 min)
	$(PY) -u -m slm.autoresearch --study hillclimb-v1

champion:  ## scale the discovered recipe up into configs/champion.yaml
	$(PY) scripts/build_champion_config.py

pretrain:  ## CRISP-DM 4: train the full model (~55 min)
	$(PY) -u -m slm.train --config configs/champion.yaml

sft:  ## CRISP-DM 4: instruction-tune the latest pretrain checkpoint into a chatbot
	$(PY) -u -m slm.sft --base-ckpt $$($(PY) -c "from slm.infer import latest_checkpoint; print(latest_checkpoint(phase='pretrain'))")

evaluate:  ## CRISP-DM 5: full evaluation report for the latest pretrain and sft checkpoints
	$(PY) -m slm.evaluate --phase pretrain --instruct data/raw/instruct_valid.txt
	$(PY) -m slm.evaluate --phase sft --instruct data/raw/instruct_valid.txt

dashboard:  ## CRISP-DM 6: serve dashboard + chat on $(PORT)
	$(PY) -m uvicorn app.server:app --host 127.0.0.1 --port $(PORT)

chat:  ## terminal chat REPL against the latest sft checkpoint
	$(PY) -m slm.infer --chat --phase sft

test:  ## run the test suite
	$(PY) -m pytest tests/ -q

all: data sft-data autoresearch champion pretrain sft evaluate  ## full pipeline, end to end

clean:  ## remove runs and processed data (keeps raw downloads)
	rm -rf runs/* data/processed/*
