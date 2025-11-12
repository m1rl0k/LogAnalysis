"""
Pytest test suite for Analytics API endpoints
Run with: pytest test_api.py -v
"""
import pytest
from app import app, data_store, model_manager


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    app.config['RATE_LIMIT_ENABLED'] = False  # Disable rate limiting for tests
    
    with app.test_client() as client:
        yield client
    
    # Cleanup after tests
    data_store.clear()
    for model in model_manager.list_models():
        model_manager.delete_model(model)


@pytest.fixture
def sample_dataset():
    """Sample dataset for testing."""
    return {
        "name": "test_data",
        "data": [
            {"sales": 100, "marketing": 20, "temperature": 15},
            {"sales": 120, "marketing": 25, "temperature": 18},
            {"sales": 95, "marketing": 15, "temperature": 12},
            {"sales": 150, "marketing": 30, "temperature": 20},
            {"sales": 200, "marketing": 50, "temperature": 22},
            {"sales": 110, "marketing": 22, "temperature": 16},
            {"sales": 130, "marketing": 28, "temperature": 19},
            {"sales": 500, "marketing": 25, "temperature": 18},  # Outlier
            {"sales": 125, "marketing": 26, "temperature": 17},
            {"sales": 140, "marketing": 29, "temperature": 21}
        ]
    }


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_health_check(self, client):
        response = client.get('/health')
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'version' in data
    
    def test_status(self, client):
        response = client.get('/status')
        assert response.status_code == 200
        data = response.get_json()
        assert 'datasets' in data
        assert 'models' in data


class TestDataManagement:
    """Test data upload and management endpoints."""
    
    def test_upload_data(self, client, sample_dataset):
        response = client.post('/upload_data', json=sample_dataset)
        assert response.status_code == 200
        data = response.get_json()
        assert data['name'] == 'test_data'
        assert data['shape']['rows'] == 10
        assert data['shape']['columns'] == 3
    
    def test_upload_data_missing_name(self, client):
        response = client.post('/upload_data', json={"data": [{"a": 1}]})
        assert response.status_code == 400
        assert 'name is required' in response.get_json()['error']
    
    def test_upload_data_missing_data(self, client):
        response = client.post('/upload_data', json={"name": "test"})
        assert response.status_code == 400
    
    def test_list_datasets(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        response = client.get('/list_datasets')
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['datasets']) > 0
        assert data['datasets'][0]['name'] == 'test_data'
    
    def test_delete_dataset(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        response = client.post('/delete_dataset', json={"dataset": "test_data"})
        assert response.status_code == 200
        assert 'deleted' in response.get_json()['message']


class TestOutlierDetection:
    """Test outlier detection endpoints."""
    
    def test_detect_outliers_iqr(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        
        response = client.post('/detect_outliers', json={
            "dataset": "test_data",
            "method": "iqr",
            "columns": ["sales"]
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['n_outliers'] > 0
        assert data['method'] == 'IQR'
        assert 'outlier_indices' in data
    
    def test_detect_outliers_zscore(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        
        response = client.post('/detect_outliers', json={
            "dataset": "test_data",
            "method": "zscore",
            "threshold": 2.0
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['method'] == 'Z-Score'
    
    def test_detect_outliers_iforest(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        
        response = client.post('/detect_outliers', json={
            "dataset": "test_data",
            "method": "iforest",
            "contamination": 0.1
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'n_outliers' in data
        assert 'outlier_score' in data
    
    def test_detect_outliers_dataset_not_found(self, client):
        response = client.post('/detect_outliers', json={
            "dataset": "nonexistent",
            "method": "iqr"
        })
        assert response.status_code == 404


class TestRegression:
    """Test regression endpoints."""
    
    def test_regression(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        
        response = client.post('/regression', json={
            "dataset": "test_data",
            "target": "sales",
            "features": ["marketing", "temperature"]
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'metrics' in data
        assert 'predictions' in data
        assert 'feature_importance' in data


class TestModelTraining:
    """Test model training and prediction endpoints."""
    
    def test_train_model(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        
        response = client.post('/train_model', json={
            "dataset": "test_data",
            "target": "sales",
            "model_name": "test_model",
            "features": ["marketing"]
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['model_saved'] is True
        assert 'metrics' in data
    
    def test_predict_with_model(self, client, sample_dataset):
        # Upload data and train model
        client.post('/upload_data', json=sample_dataset)
        client.post('/train_model', json={
            "dataset": "test_data",
            "target": "sales",
            "model_name": "predictor"
        })
        
        # Make predictions
        response = client.post('/predict', json={
            "model_name": "predictor",
            "data": [{"marketing": 30, "temperature": 20}]
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'predictions' in data
        assert len(data['predictions']) == 1
    
    def test_list_models(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        client.post('/train_model', json={
            "dataset": "test_data",
            "target": "sales",
            "model_name": "model1"
        })
        
        response = client.get('/list_models')
        assert response.status_code == 200
        data = response.get_json()
        assert 'model1' in data['models']


class TestForecasting:
    """Test forecasting endpoints."""
    
    def test_forecast_linear(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        
        response = client.post('/forecast', json={
            "dataset": "test_data",
            "value_column": "sales",
            "method": "linear",
            "horizon": 5
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['forecast']) == 5
        assert 'metrics' in data
    
    def test_forecast_exponential(self, client, sample_dataset):
        client.post('/upload_data', json=sample_dataset)
        
        response = client.post('/forecast', json={
            "dataset": "test_data",
            "value_column": "sales",
            "method": "exponential",
            "horizon": 3,
            "alpha": 0.3
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert len(data['forecast']) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

