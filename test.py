#!/usr/bin/env python3
"""
Simple CLI for Analytics API - train and analyze workflow
"""
import os
import requests
import json
import sys
import base64
import argparse

BASE_URL = os.environ.get('ANALYTICS_API_URL', 'http://localhost:5000')


def check_health():
    """Check API health."""
    print("=== Health Check ===")
    response = requests.get(f'{BASE_URL}/health')
    print(json.dumps(response.json(), indent=2))
    print()


def upload_sample_data():
    """Upload sample dataset."""
    print("=== Uploading Sample Dataset ===")

    # Sample sales data
    data = {
        "name": "sales_data",
        "data": [
            {"date": "2024-01-01", "sales": 100, "marketing": 20, "temperature": 15},
            {"date": "2024-01-02", "sales": 120, "marketing": 25, "temperature": 18},
            {"date": "2024-01-03", "sales": 95, "marketing": 15, "temperature": 12},
            {"date": "2024-01-04", "sales": 150, "marketing": 30, "temperature": 20},
            {"date": "2024-01-05", "sales": 200, "marketing": 50, "temperature": 22},
            {"date": "2024-01-06", "sales": 110, "marketing": 22, "temperature": 16},
            {"date": "2024-01-07", "sales": 130, "marketing": 28, "temperature": 19},
            {"date": "2024-01-08", "sales": 500, "marketing": 25, "temperature": 18},  # Outlier
            {"date": "2024-01-09", "sales": 125, "marketing": 26, "temperature": 17},
            {"date": "2024-01-10", "sales": 140, "marketing": 29, "temperature": 21},
        ]
    }

    response = requests.post(f'{BASE_URL}/upload_data', json=data)
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    print()


def detect_outliers():
    """Detect outliers in the dataset."""
    print("=== Detecting Outliers (IQR method) ===")

    payload = {
        "dataset": "sales_data",
        "method": "iqr",
        "columns": ["sales", "marketing"]
    }

    response = requests.post(f'{BASE_URL}/detect_outliers', json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Found {result.get('n_outliers', 0)} outliers")
    print(f"Outlier indices: {result.get('outlier_indices', [])}")
    print()


def train_regression_model():
    """Train a regression model."""
    print("=== Training Regression Model ===")

    payload = {
        "dataset": "sales_data",
        "target": "sales",
        "features": ["marketing", "temperature"],
        "model_name": "sales_predictor",
        "test_size": 0.3
    }

    response = requests.post(f'{BASE_URL}/train_model', json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()

    if response.status_code == 200:
        print(f"Model trained successfully!")
        print(f"Test R²: {result['metrics']['test_r2']:.3f}")
        print(f"Test RMSE: {result['metrics']['test_rmse']:.3f}")
        print("\nTop Features:")
        for feat in result['feature_importance'][:3]:
            print(f"  {feat['feature']}: {feat['coefficient']:.3f}")
    else:
        print(f"Error: {result}")
    print()


def make_predictions():
    """Make predictions with trained model."""
    print("=== Making Predictions ===")

    # New data to predict
    payload = {
        "model_name": "sales_predictor",
        "data": [
            {"marketing": 35, "temperature": 23},
            {"marketing": 40, "temperature": 25},
            {"marketing": 20, "temperature": 14}
        ]
    }

    response = requests.post(f'{BASE_URL}/predict', json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()

    if response.status_code == 200:
        print(f"Predictions: {result['predictions']}")
    else:
        print(f"Error: {result}")
    print()


def forecast_time_series():
    """Forecast future values."""
    print("=== Forecasting (Linear Trend) ===")

    payload = {
        "dataset": "sales_data",
        "value_column": "sales",
        "method": "linear",
        "horizon": 5
    }

    response = requests.post(f'{BASE_URL}/forecast', json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()

    if response.status_code == 200:
        print(f"Forecast for next 5 periods: {result['forecast']}")
        print(f"MAE: {result['metrics']['mae']:.3f}")
        print(f"Trend slope: {result['metrics']['slope']:.3f}")
    else:
        print(f"Error: {result}")
    print()


def list_all():
    """List datasets and models."""
    print("=== Listing Datasets ===")
    response = requests.get(f'{BASE_URL}/list_datasets')
    print(json.dumps(response.json(), indent=2))
    print()

    print("=== Listing Models ===")
    response = requests.get(f'{BASE_URL}/list_models')
    print(json.dumps(response.json(), indent=2))
    print()


def simple_train(file_path, target=None):
    """Simple train workflow."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {file_path}")
    print('='*60)

    response = requests.post(f'{BASE_URL}/train', json={
        'file': file_path,
        'target': target
    })

    if response.status_code == 200:
        result = response.json()
        print(f"✓ Dataset: {result['dataset']}")
        print(f"✓ Rows: {result['rows']}, Columns: {result['columns']}")
        print(f"✓ Numeric columns: {', '.join(result['numeric_columns'])}")

        if result['model_trained']:
            print(f"\n✓ Model trained!")
            print(f"  R²: {result['model_metrics']['test_r2']:.3f}")
            print(f"  RMSE: {result['model_metrics']['test_rmse']:.3f}")
    else:
        print(f"✗ Error: {response.json()}")
        return False

    return True


def simple_analyze(file_path, target=None):
    """Simple analyze workflow with visualization."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {file_path}")
    print('='*60)

    response = requests.post(f'{BASE_URL}/analyze', json={
        'file': file_path,
        'target': target
    })

    if response.status_code == 200:
        result = response.json()

        # Summary
        print(f"\n📊 SUMMARY")
        print(f"  Dataset: {result['dataset']}")
        print(f"  Rows: {result['summary']['rows']}")
        print(f"  Columns: {result['summary']['columns']}")

        # Outliers
        print(f"\n🔍 OUTLIERS ({result['outliers']['method']})")
        print(f"  Found: {result['outliers']['count']} ({result['outliers']['percentage']:.1f}%)")
        if result['outliers']['indices']:
            print(f"  Indices: {result['outliers']['indices']}")

        # Anomalies
        print(f"\n⚠️  ANOMALIES ({result['anomalies']['method']})")
        print(f"  Found: {result['anomalies']['count']} ({result['anomalies']['percentage']:.1f}%)")

        # Forecasts
        print(f"\n📈 FORECASTS")
        print(f"  Linear Trend:")
        print(f"    Next 5: {[f'{x:.1f}' for x in result['forecasts']['linear_trend']['values']]}")
        print(f"    MAE: {result['forecasts']['linear_trend']['mae']:.2f}")
        print(f"    Slope: {result['forecasts']['linear_trend']['slope']:.3f}")

        print(f"  Exponential Smoothing:")
        print(f"    Next 5: {[f'{x:.1f}' for x in result['forecasts']['exponential_smoothing']['values']]}")
        print(f"    MAE: {result['forecasts']['exponential_smoothing']['mae']:.2f}")

        # Regression
        if 'regression' in result:
            print(f"\n📉 REGRESSION (target: {result['regression']['target']})")
            print(f"  R²: {result['regression']['r2']:.3f}")
            print(f"  RMSE: {result['regression']['rmse']:.3f}")
            print(f"  Top Features:")
            for feat in result['regression']['top_features']:
                print(f"    {feat['feature']}: {feat['coefficient']:.3f}")

        # Visualization
        if 'visualization_path' in result:
            print(f"\n📸 Visualization saved: {result['visualization_path']}")

        print(f"\n{'='*60}")
        print("✓ Analysis complete!")
        print('='*60)
    else:
        print(f"✗ Error: {response.json()}")
        return False

    return True


def main():
    """Run simple workflow demo."""
    parser = argparse.ArgumentParser(description='Analytics API CLI')
    parser.add_argument('--train', action='store_true', help='Run training workflow')
    parser.add_argument('--analyze', action='store_true', help='Run analysis workflow')
    parser.add_argument('--file', type=str, help='Data file path')
    parser.add_argument('--target', type=str, help='Target column for regression')
    args = parser.parse_args()

    try:
        # Check health
        response = requests.get(f'{BASE_URL}/health')
        if response.status_code != 200:
            print("ERROR: API not healthy")
            sys.exit(1)

        if args.train and args.file:
            simple_train(args.file, args.target)
        elif args.analyze and args.file:
            simple_analyze(args.file, args.target)
        else:
            # Run demo on all mock data
            print("Analytics API - Simple Workflow Demo")
            print("="*60)

            files = [
                ('mock_data/sales_data.json', 'sales'),
                ('mock_data/sensor_data.json', 'temperature'),
                ('mock_data/stock_prices.json', 'close')
            ]

            for file_path, target in files:
                simple_analyze(file_path, target)

            print("\n" + "="*60)
            print("✓ All demos completed!")
            print("="*60)

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to API. Make sure the server is running:")
        print("  python app.py")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
