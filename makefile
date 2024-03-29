.DEFAULT_GOAL := help
.PHONY: help
.EXPORT_ALL_VARIABLES:

include .env
help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 _]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

# Code
deps: # Install deps
	pip install -e .[all]
pre: # Run pre-commit hooks on all files
	pre-commit run --all-files
cov: # Compute coverage
	pytest --cov=src --cov-report term-missing

# Documentation
serve: # Run local doc
	mkdocs serve
deploy_doc: # Deploy doc on github pages
	mkdocs gh-deploy
