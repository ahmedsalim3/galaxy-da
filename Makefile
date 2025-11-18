install:
	git submodule update --init --recursive
	uv sync --all-extras

install-dev:
	git submodule update --init --recursive
	uv sync --all-extras
	uv run pre-commit install

fix:
	uv run pre-commit run --all-files

test:
	uv run pytest --cov=PACKAGE --cov-report=term-missing

clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
	rm -rf .pytest_cache .mypy_cache .ruff_cache/ .coverage htmlcov dist build *.egg-info

lint:
	uv run black .
	uv run isort .

build-docs:
	cd docs && uv run mkdocs build

serve-docs:
	cd docs && uv run mkdocs serve

deploy-docs:
	cd docs && uv run mkdocs gh-deploy --force