format: 
	black tumortwin
	isort tumortwin

	black tests
	isort tests

lint:
	flake8 tumortwin
	mypy tumortwin

	flake8 tests
	mypy tests

test:
	pytest tests

docs-build:
	mkdocs build

docs-serve:
	mkdocs serve

help: ## See a list of all available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.* ?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.PHONY: all $(MAKECMDGOALS)

.DEFAULT_GOAL := help