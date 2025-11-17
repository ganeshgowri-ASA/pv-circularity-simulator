.PHONY: help install install-dev test lint format type-check clean run-ui run-example

help:
	@echo "PV Circularity Simulator - System Validation Module"
	@echo ""
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make install-dev    - Install with development dependencies"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make lint           - Run linter (ruff)"
	@echo "  make format         - Format code (black)"
	@echo "  make type-check     - Run type checker (mypy)"
	@echo "  make clean          - Clean up generated files"
	@echo "  make run-ui         - Launch Streamlit UI"
	@echo "  make run-example    - Run basic validation example"

install:
	pip install -r requirements.txt

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov=ui --cov-report=html --cov-report=term-missing

lint:
	ruff check src/ ui/ tests/

format:
	black src/ ui/ tests/ examples/

type-check:
	mypy src/ ui/

clean:
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf exports/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

run-ui:
	streamlit run ui/system_validation_ui.py

run-example:
	python examples/basic_validation_example.py

all: format lint type-check test
