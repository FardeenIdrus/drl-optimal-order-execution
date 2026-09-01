# Quality gates and common tasks. Run from the repository root.
PY := .venv/bin

.PHONY: install test lint verify-archive

install:
	python3.12 -m venv .venv
	$(PY)/pip install -r requirements.txt
	$(PY)/pip install -e ".[dev]"

test:
	$(PY)/pytest tests -q

lint:
	$(PY)/ruff check src tests

verify-archive:
	cd results_archive && shasum -a 256 -c CHECKSUMS.sha256 | grep -v 'OK$$' ; true
