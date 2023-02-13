pre-commit:
	black . && \
	ruff --fix src/ && \
	mypy --ignore-missing-imports src/