"""
Pytest test suite for Analytics API
Run with: pytest test_analytics.py -v
"""
import pytest
import pandas as pd
import numpy as np
from app import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_anomalies_iforest,
    linear_regression,
    forecast_exponential_smoothing,
    forecast_linear_trend,
    data_store,
    model_manager
)


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'sales': [100, 120, 95, 150, 200, 110, 130, 500, 125, 140],  # 500 is outlier
        'marketing': [20, 25, 15, 30, 50, 22, 28, 25, 26, 29],
        'temperature': [15, 18, 12, 20, 22, 16, 19, 18, 17, 21]
    })


@pytest.fixture
def time_series_data():
    """Create time series dataset."""
    return pd.DataFrame({
        'value': [10, 12, 15, 14, 18, 20, 22, 25, 28, 30]
    })


class TestOutlierDetection:
    """Test outlier detection algorithms."""
    
    def test_iqr_detects_outliers(self, sample_data):
        result = detect_outliers_iqr(sample_data, ['sales'])
        assert result['n_outliers'] > 0
        assert result['method'] == 'IQR'
        assert len(result['is_outlier']) == len(sample_data)
        assert result['outlier_percentage'] > 0
    
    def test_zscore_detects_outliers(self, sample_data):
        result = detect_outliers_zscore(sample_data, ['sales'], threshold=2.0)
        assert result['n_outliers'] > 0
        assert result['method'] == 'Z-Score'
        assert result['threshold'] == 2.0
        assert len(result['outlier_score']) == len(sample_data)
    
    def test_iforest_detects_anomalies(self, sample_data):
        result = detect_anomalies_iforest(sample_data, ['sales', 'marketing'], contamination=0.1)
        assert result['n_anomalies'] > 0
        assert result['method'] == 'Isolation Forest'
        assert result['contamination'] == 0.1
        assert len(result['is_anomaly']) == len(sample_data)
    
    def test_outlier_detection_with_no_numeric_columns(self):
        df = pd.DataFrame({'text': ['a', 'b', 'c']})
        with pytest.raises(ValueError, match="No numeric columns found"):
            detect_outliers_iqr(df)


class TestLinearRegression:
    """Test linear regression functionality."""
    
    def test_regression_basic(self, sample_data):
        result = linear_regression(sample_data, 'sales', ['marketing', 'temperature'], test_size=0.3)
        
        assert 'metrics' in result
        assert 'test_r2' in result['metrics']
        assert 'test_rmse' in result['metrics']
        assert 'feature_importance' in result
        assert len(result['feature_importance']) == 2
    
    def test_regression_auto_features(self, sample_data):
        result = linear_regression(sample_data, 'sales', feature_cols=None, test_size=0.2)
        assert len(result['feature_importance']) > 0
    
    def test_regression_saves_model(self, sample_data):
        result = linear_regression(sample_data, 'sales', ['marketing'], model_name='test_model')
        assert result['model_saved'] is True
        assert result['model_name'] == 'test_model'
        
        # Verify model was saved
        saved_model = model_manager.get_regression_model('test_model')
        assert saved_model is not None
        
        # Cleanup
        model_manager.delete_model('test_model')
    
    def test_regression_invalid_target(self, sample_data):
        with pytest.raises(ValueError, match="Target column"):
            linear_regression(sample_data, 'nonexistent', ['marketing'])
    
    def test_regression_no_features(self, sample_data):
        df = pd.DataFrame({'text': ['a', 'b', 'c'], 'target': [1, 2, 3]})
        with pytest.raises(ValueError, match="No feature columns"):
            linear_regression(df, 'target')


class TestForecasting:
    """Test forecasting algorithms."""
    
    def test_exponential_smoothing(self, time_series_data):
        result = forecast_exponential_smoothing(time_series_data, 'value', horizon=5, alpha=0.3)
        
        assert len(result['forecast']) == 5
        assert 'metrics' in result
        assert 'mae' in result['metrics']
        assert result['method'] == 'Exponential Smoothing'
        assert len(result['smoothed_values']) == len(time_series_data)
    
    def test_linear_trend(self, time_series_data):
        result = forecast_linear_trend(time_series_data, 'value', horizon=5)
        
        assert len(result['forecast']) == 5
        assert 'metrics' in result
        assert 'slope' in result['metrics']
        assert 'intercept' in result['metrics']
        assert result['method'] == 'Linear Trend'
    
    def test_forecast_invalid_column(self, time_series_data):
        with pytest.raises(ValueError, match="not found"):
            forecast_exponential_smoothing(time_series_data, 'nonexistent', horizon=5)
    
    def test_forecast_insufficient_data(self):
        df = pd.DataFrame({'value': [10]})
        with pytest.raises(ValueError, match="at least 2 data points"):
            forecast_linear_trend(df, 'value', horizon=5)


class TestDataStore:
    """Test data storage functionality."""
    
    def test_add_and_get_dataset(self, sample_data):
        info = data_store.add_dataset('test_ds', sample_data)
        
        assert info.name == 'test_ds'
        assert info.shape == sample_data.shape
        assert len(info.numeric_cols) == 3
        
        retrieved = data_store.get_dataset('test_ds')
        assert retrieved is not None
        assert len(retrieved) == len(sample_data)
        
        # Cleanup
        data_store.delete_dataset('test_ds')
    
    def test_list_datasets(self, sample_data):
        data_store.add_dataset('ds1', sample_data)
        data_store.add_dataset('ds2', sample_data)
        
        datasets = data_store.list_datasets()
        assert 'ds1' in datasets
        assert 'ds2' in datasets
        
        # Cleanup
        data_store.delete_dataset('ds1')
        data_store.delete_dataset('ds2')
    
    def test_delete_dataset(self, sample_data):
        data_store.add_dataset('temp_ds', sample_data)
        success = data_store.delete_dataset('temp_ds')
        assert success is True
        
        retrieved = data_store.get_dataset('temp_ds')
        assert retrieved is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

