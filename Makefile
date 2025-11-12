.PHONY: help install data server test analyze clean all

# Default target
help:
	@echo "Analytics API - Available Commands"
	@echo "===================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install all dependencies"
	@echo "  make data             - Generate large test datasets"
	@echo ""
	@echo "Running:"
	@echo "  make server           - Start the API server"
	@echo "  make dev              - Start server in development mode"
	@echo ""
	@echo "Testing & Analysis:"
	@echo "  make test             - Run all tests"
	@echo "  make analyze          - Run analysis on all datasets"
	@echo "  make analyze-logs     - Analyze server logs (10K rows)"
	@echo "  make analyze-sales    - Analyze sales data (5K rows)"
	@echo "  make analyze-sensors  - Analyze sensor data (20K rows)"
	@echo "  make analyze-weather  - Analyze weather data (3.6K rows)"
	@echo "  make analyze-stocks   - Analyze stock prices (1.4K rows)"
	@echo "  make analyze-ecom     - Analyze e-commerce data (15K rows)"
	@echo ""
	@echo "Model Management (UNIFIED MODEL):"
	@echo "  make train            - Train unified model on ALL datasets (~55K samples)"
	@echo "  make models-list      - List saved models"
	@echo "  make models-clean     - Remove all saved models"
	@echo ""
	@echo "Note: All datasets train ONE unified model that grows continuously!"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove generated files"
	@echo "  make clean-all        - Remove everything (data, models, analysis)"
	@echo ""
	@echo "Shortcuts:"
	@echo "  make all              - Install, generate data, and run server"

# Installation
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete"

# Generate datasets
data:
	@echo "Generating large test datasets..."
	python3 generate_datasets.py
	@echo "✓ Datasets created in data/ directory"

# Start server
server:
	@echo "Starting Analytics API server on port 5001..."
	PORT=5001 python app.py

dev:
	@echo "Starting server in development mode..."
	FLASK_ENV=development DEBUG=True PORT=5001 python app.py

# Testing
test:
	@echo "Running tests..."
	pytest tests/test_analytics.py -v

# Analysis commands (using CLI - no curl needed!)
analyze: analyze-logs analyze-sales analyze-sensors analyze-weather analyze-stocks analyze-ecom
	@echo ""
	@echo "✓ All datasets analyzed!"

analyze-logs:
	@python app.py analyze data/server_logs.csv response_time_ms

analyze-sales:
	@python app.py analyze data/sales_data.csv sales

analyze-sensors:
	@python app.py analyze data/sensor_data.csv temperature

analyze-weather:
	@python app.py analyze data/weather_data.csv temperature

analyze-stocks:
	@python app.py analyze data/stock_prices.csv close

analyze-ecom:
	@python app.py analyze data/ecommerce_transactions.csv amount

# Training - ALL data trains ONE unified model (using CLI!)
train:
	@echo "Training UNIFIED MODEL on all datasets..."
	@echo ""
	@python app.py train data/sales_data.csv sales
	@python app.py train data/server_logs.csv response_time_ms
	@python app.py train data/sensor_data.csv temperature
	@python app.py train data/weather_data.csv temperature
	@python app.py train data/stock_prices.csv close
	@python app.py train data/ecommerce_transactions.csv amount
	@echo ""
	@echo "✓ UNIFIED MODEL trained on ~55K total samples!"

# Model management
models-list:
	@echo "Saved models:"
	@ls -lh models/ 2>/dev/null || echo "  No models found"

models-clean:
	@echo "Removing all saved models..."
	rm -rf models/*.joblib
	@echo "✓ Models cleaned"

# Cleanup
clean:
	@echo "Cleaning up generated files..."
	rm -rf __pycache__ .pytest_cache *.pyc
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleanup complete"

clean-all: clean models-clean
	@echo "Removing all generated data..."
	rm -rf analysis/*.png
	@echo "✓ Full cleanup complete"

# All-in-one setup
all: install data
	@echo ""
	@echo "✓ Setup complete! Starting server..."
	@echo ""
	$(MAKE) server

