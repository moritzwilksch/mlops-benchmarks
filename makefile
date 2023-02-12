pre-commit:
	black . && \
	ruff --fix --ignore=E501 src/