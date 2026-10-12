PYTHON := uv run python
MAIN := src/main.py

.PHONY: install run debug clean lint lint-strict


install: 
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) $(MAIN)

debug:
	$(PYTHON) -m pdb $(MAIN)

clean:
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -empty -delete
	find . -type d -name '.mypy_cache' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +

lint:
	flake8 .
	mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs \
		--check-untyped-defs
