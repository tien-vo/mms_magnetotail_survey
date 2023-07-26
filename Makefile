SHELL = /bin/zsh -i
NAME := mms_magnetotail_survey
MAMBA := $(shell command -v micromamba 2> /dev/null)
INSTALL_STAMP := .install.stamp
CONDA_LOCK := conda-linux-64.lock
POETRY_LOCK := poetry.lock


.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo "Edit help string"
	@echo $(POETRY)

install: $(INSTALL_STAMP)
$(INSTALL_STAMP): pyproject.toml $(POETRY_LOCK) $(CONDA_LOCK)
	@if [ -z $(MAMBA) ]; then \
	echo "Micromamba not found. See https://mamba.readthedocs.io/en/latest/installation.html for installation."; \
	exit 1; \
	fi
	@echo "Creating virtual environment from $(CONDA_LOCK) ..."
	@micromamba create --quiet --yes --override-channels --name ${NAME} --file conda-linux-64.lock
	@echo "Installing packages from $(POETRY_LOCK) ..."
	@micromamba run -n ${NAME} poetry install
	@micromamba run -n ${NAME} pip install spacepy --no-build-isolation
	@touch $(INSTALL_STAMP)
	@echo "Done installation!"

.PHONY: clean
clean:
	find . -type d -name "__pycache__" | xargs rm -rf {};
	rm -rf $(INSTALL_STAMP)
