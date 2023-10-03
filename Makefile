NAME := mms_survey
MAMBA := $(shell command -v micromamba 2> /dev/null)
CONDA_LOCK := conda-osx-64.lock
POETRY_LOCK := poetry.lock

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "TODO: Edit help string"
	@echo $(POETRY)

.PHONY: install
install: pyproject.toml $(POETRY_LOCK) $(CONDA_LOCK)
	@if [ -z $(MAMBA) ]; then \
	echo "Micromamba not found. See https://mamba.readthedocs.io/en/latest/installation.html for installation."; \
	exit 1; \
	fi
	@echo "Creating virtual environment from $(CONDA_LOCK) ..."
	@micromamba create --quiet --yes --override-channels --name ${NAME} --file conda-linux-64.lock
	@echo "Installing packages from $(POETRY_LOCK) ..."
	@micromamba run -n ${NAME} poetry install
	@echo "Done installation!"

.PHONY: format
format:
	poetry run isort src/
	poetry run black src/

.PHONY: clean
clean:
	find . -type d -name "__pycache__" | xargs rm -rf {};
