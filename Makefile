.DEFAULT_GOAL := help

.PHONY: help setup format lint types test check clean

help:
	@echo "Available commands:"
	@echo "  make setup   Install the project and development dependencies"
	@echo "  make format  Format the source and tests"
	@echo "  make lint    Run Ruff checks"
	@echo "  make types   Run type checks"
	@echo "  make test    Run tests"
	@echo "  make check   Run lint, type checks, and tests"
	@echo "  make clean   Remove generated caches and build artifacts"

setup:
	uv sync

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

types:
	uv run ty check

test:
	uv run pytest

check: lint types test

clean:
	uv run python -c "import shutil; from pathlib import Path; [shutil.rmtree(p, ignore_errors=True) for name in ('__pycache__', '.pytest_cache', '.ruff_cache', 'build', 'dist') for p in Path('.').rglob(name)]"
