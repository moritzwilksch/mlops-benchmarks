pre-commit:
	black . && \
	ruff --fix src/ && \
	mypy src/