.PHONY: install data train eval test lint clean format

install:
	pip install -r requirements-dev.txt
	pip install -e .

# Regenerate the synthetic sample data (data/raw/churn_data.csv already
# ships a committed copy, this is only needed if you want a different
# size or seed).
data:
	python scripts/make_sample_data.py --output data/raw/churn_data.csv

train:
	PYTHONPATH=src python src/train.py --config configs/model_config.yaml --input data/raw/churn_data.csv

eval:
	PYTHONPATH=src python src/predict.py --input data/raw/churn_data.csv --output predictions.csv

test:
	pytest tests/ -v --tb=short

lint:
	flake8 . --max-line-length=120
	mypy . --ignore-missing-imports

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	rm -rf .pytest_cache .mypy_cache htmlcov .coverage

format:
	black .
	isort .
