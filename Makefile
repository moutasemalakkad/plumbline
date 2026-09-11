.PHONY: help install test example example-routing clean

# Force public PyPI. In environments where pip/uv default to a private index,
# the install 401s on public packages like pytest. --no-project on run/test
# avoids uv locking the optional cloud extras (bigquery/gemini) for the fast
# path.
INDEX := https://pypi.org/simple/
# Pin the interpreter. With another project's venv active, VIRTUAL_ENV wins over
# ./.venv and uv installs this package into that unrelated venv instead.
export VIRTUAL_ENV := $(CURDIR)/.venv

help: ## show available targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## create .venv and install the package + dev deps (no cloud extras)
	uv venv --allow-existing
	uv pip install --index-url $(INDEX) -e ".[dev]"

test: ## run the fast test suite (pure stdlib + fakes, no cloud)
	uv run --no-project pytest -q

example: ## build the example SQLite DB and print score + gate reports
	uv run --no-project python examples/build_db.py
	uv run --no-project python -m plumbline score \
		--db examples/demo.db --predictions examples/predictions.csv
	# `-` so make ignores the gate's non-zero exit: the example golds trip
	# defects on purpose, and the point is to print them, not fail the demo.
	-uv run --no-project python -m plumbline gate \
		--db examples/demo.db --golds examples/golds.csv

example-routing: ## score the routing example twice: without and with the catalog snap
	# Without a catalog the typo in r5 is a plain miss; with one it snaps and
	# the report says so. Both runs to show the flag changes the number.
	uv run --no-project python -m plumbline score-routing \
		--predictions examples/routing_predictions.csv
	@echo
	uv run --no-project python -m plumbline score-routing \
		--predictions examples/routing_predictions.csv \
		--catalog examples/routing_catalog.txt

clean: ## remove venv, caches, and scratch example DBs
	rm -rf .venv .pytest_cache **/__pycache__ *.egg-info examples/*.db
