# One command per step. Run from the repository root. Outputs go outside the repository.
#
#   make run TEAM=<team_name>     dead ball -> H / A -> $(OUT)/<team_name>.csv   (needs MTAH_DATA_DIR)
#   make reference                degraded Sample Game 3 + event-derived labels -> $(REF)
#   make evaluate                 model vs rule vs baseline on the reference set -> $(EVAL)
#   make train                    refit both models from open data (rewrites the two JSON files)
#   make test                     unit tests on synthetic fixtures

PY          ?= python3
OUT         ?= $(HOME)/mtah_out/submission
REF         ?= $(HOME)/mtah_out/reference
EVAL        ?= $(HOME)/mtah_out/evaluate
SAMPLE_DATA ?= $(HOME)/sample-data
TEAM        ?=

export PYTHONPATH := $(CURDIR):$(CURDIR)/src

.PHONY: run reference evaluate train test

run:
	@test -n "$(TEAM)" || (echo "usage: make run TEAM=<team_name>"; exit 2)
	$(PY) -m pipeline.make_submission $(OUT) --team $(TEAM)

reference:
	$(PY) -m possession.reference $(REF) --sample-data $(SAMPLE_DATA) --games 3

evaluate:
	$(PY) -m pipeline.evaluate_chain $(EVAL) --reference $(REF)

train:
	$(PY) -m deadball.train_players $(SAMPLE_DATA)
	$(PY) -m possession.train $(REF) --write

test:
	$(PY) -m pytest -q -p no:cacheprovider src/possession/tests src/pipeline/tests
