# Common tasks. CPU-only targets work anywhere; `run-*` targets need a CUDA GPU.
PY ?= python3
VENV ?= .venv
BIN = $(VENV)/bin

.PHONY: venv install install-train install-sarvam test lint smoke data compare run-qwen run-sarvam clean

venv:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip

install: venv             ## CPU: data, metrics, reports, tests
	$(BIN)/pip install -e ".[dev]"

install-train: venv       ## GPU: Qwen and most models
	$(BIN)/pip install -e ".[train,dev]"
	-$(BIN)/pip uninstall -y torchao

install-sarvam: venv      ## GPU: Sarvam-30B (transformers 4.x)
	$(BIN)/pip install -e ".[sarvam,dev]"
	-$(BIN)/pip uninstall -y torchao

test:
	$(BIN)/pytest

lint:
	$(BIN)/ruff check src tests scripts

smoke:                    ## whole pipeline on CPU with a stand-in model, in seconds
	$(BIN)/zameen smoke

data:                     ## build and export the datasets for the public sample
	$(BIN)/zameen data -c configs/qwen2.5-14b-medium.yaml --export runs/data

compare:                  ## rebuild results/comparison.md from the committed runs
	$(BIN)/zameen compare results/qwen2.5-14b-small results/sarvam-30b-small results/qwen2.5-14b-medium results/sarvam-30b-medium --out results

run-qwen:
	$(BIN)/zameen run -c configs/qwen2.5-14b-medium.yaml

run-sarvam:
	$(BIN)/zameen run -c configs/sarvam-30b-medium.yaml

clean:
	rm -rf runs .pytest_cache .ruff_cache src/*.egg-info
