"""
Production-ready Analytics API with ML-based anomaly detection, forecasting, and regression.
Handles any tabular data: CSV, Excel, JSON, logs.
"""
from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge, ElasticNet, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from scipy import stats
import re
from collections import deque
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for production
import matplotlib.pyplot as plt
import io
import base64
import joblib
import os
import logging
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Union
from threading import Lock
from functools import wraps
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables
load_dotenv()

# Import configuration
from config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE) if config.LOG_FILE else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try to import Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
    logger.info("Prophet loaded successfully")
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not available. Install with: pip install prophet")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

# Enable CORS
CORS(app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=[f"{config.RATE_LIMIT_PER_MINUTE}/minute", f"{config.RATE_LIMIT_PER_HOUR}/hour"],
    enabled=config.RATE_LIMIT_ENABLED
)

# Initialize Prometheus metrics
if config.ENABLE_METRICS:
    metrics = PrometheusMetrics(app)
    metrics.info('log_analysis_api', 'Log Analysis API', version=config.MODEL_VERSION)

# Thread-safe data structures
@dataclass
class DatasetInfo:
    """Metadata about a stored dataset."""
    name: str
    shape: Tuple[int, int]
    columns: List[str]
    dtypes: Dict[str, str]
    timestamp: datetime
    numeric_cols: List[str]
    categorical_cols: List[str]
    datetime_cols: List[str]


class DataStore:
    """Thread-safe data storage with memory management."""

    def __init__(self, max_datasets: int = 10):
        self.max_datasets = max_datasets
        self.datasets: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, DatasetInfo] = {}
        self.lock = Lock()
        logger.info(f"Initialized DataStore with max_datasets={max_datasets}")

    def add_dataset(self, name: str, df: pd.DataFrame) -> DatasetInfo:
        """Add dataset to the store with thread safety."""
        with self.lock:
            # Enforce limit
            if len(self.datasets) >= self.max_datasets and name not in self.datasets:
                # Remove oldest dataset
                oldest = min(self.metadata.items(), key=lambda x: x[1].timestamp)
                del self.datasets[oldest[0]]
                del self.metadata[oldest[0]]
                logger.info(f"Removed oldest dataset: {oldest[0]}")

            # Analyze columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

            # Store dataset and metadata
            self.datasets[name] = df
            info = DatasetInfo(
                name=name,
                shape=df.shape,
                columns=df.columns.tolist(),
                dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
                timestamp=datetime.now(),
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                datetime_cols=datetime_cols
            )
            self.metadata[name] = info

            logger.info(f"Added dataset '{name}': {df.shape[0]} rows, {df.shape[1]} columns")
            return info

    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get dataset from the store."""
        with self.lock:
            return self.datasets.get(name)

    def get_metadata(self, name: str) -> Optional[DatasetInfo]:
        """Get dataset metadata."""
        with self.lock:
            return self.metadata.get(name)

    def list_datasets(self) -> List[str]:
        """List all stored datasets."""
        with self.lock:
            return list(self.datasets.keys())

    def delete_dataset(self, name: str) -> bool:
        """Delete a dataset."""
        with self.lock:
            if name in self.datasets:
                del self.datasets[name]
                del self.metadata[name]
                logger.info(f"Deleted dataset: {name}")
                return True
            return False

    def clear(self) -> None:
        """Clear all datasets."""
        with self.lock:
            self.datasets.clear()
            self.metadata.clear()
            logger.info("Cleared all datasets from store")


# Global data store
data_store = DataStore(max_datasets=10)

# Model storage
class ModelManager:
    """Thread-safe model management for regression and forecasting."""

    def __init__(self):
        self.regression_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.lock = Lock()
        logger.info("Initialized ModelManager")

    def save_regression_model(self, name: str, model: Any, scaler: Optional[StandardScaler] = None) -> None:
        """Save a regression model."""
        with self.lock:
            self.regression_models[name] = model
            if scaler:
                self.scalers[name] = scaler
            logger.info(f"Saved regression model: {name}")

    def get_regression_model(self, name: str) -> Optional[Tuple[Any, Optional[StandardScaler]]]:
        """Get a regression model and its scaler."""
        with self.lock:
            model = self.regression_models.get(name)
            scaler = self.scalers.get(name)
            return (model, scaler) if model else None

    def list_models(self) -> List[str]:
        """List all saved models."""
        with self.lock:
            return list(self.regression_models.keys())

    def delete_model(self, name: str) -> bool:
        """Delete a model."""
        with self.lock:
            if name in self.regression_models:
                del self.regression_models[name]
                if name in self.scalers:
                    del self.scalers[name]
                logger.info(f"Deleted model: {name}")
                return True
            return False

model_manager = ModelManager()


# Utility decorators
def track_time(func):
    """Decorator to track execution time of functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} completed in {elapsed:.3f}s")
        return result
    return wrapper


def validate_request(required_fields: List[str]):
    """Decorator to validate request JSON fields."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not request.json:
                logger.warning(f"Request to {func.__name__} missing JSON body")
                return jsonify({'error': 'Request must be JSON'}), 400

            missing_fields = [field for field in required_fields if field not in request.json]
            if missing_fields:
                logger.warning(f"Request to {func.__name__} missing fields: {missing_fields}")
                return jsonify({'error': f'Missing required fields: {missing_fields}'}), 400

            return func(*args, **kwargs)
        return wrapper
    return decorator


# ==================== CORE ANALYTICS FUNCTIONS ====================

# ==================== LOG PARSING & TEXT ANOMALY DETECTION ====================

def parse_log_file(file_path: str) -> pd.DataFrame:
    """
    Parse raw log files into structured data.
    Handles common log formats and extracts features.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        log_data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Extract timestamp
            timestamp_match = re.search(r'\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}', line)
            timestamp = timestamp_match.group(0) if timestamp_match else None

            # Extract log level
            level_match = re.search(r'\b(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\b', line, re.IGNORECASE)
            level = level_match.group(0).upper() if level_match else 'INFO'

            # Extract numeric values
            numbers = re.findall(r'\b\d+\.?\d*\b', line)

            log_data.append({
                'raw_message': line,
                'timestamp': timestamp,
                'level': level,
                'message_length': len(line),
                'word_count': len(line.split()),
                'numeric_count': len(numbers),
                'has_error': 1 if 'error' in line.lower() or 'fail' in line.lower() else 0,
                'has_warning': 1 if 'warn' in line.lower() else 0
            })

        df = pd.DataFrame(log_data)
        logger.info(f"Parsed {len(df)} log lines")
        return df

    except Exception as e:
        logger.error(f"Error parsing log file: {e}", exc_info=True)
        raise


@track_time
def detect_log_anomalies_tfidf(log_messages: List[str],
                               eps: float = 0.5,
                               min_samples: int = 2,
                               max_features: int = 100) -> Dict[str, Any]:
    """
    Detect anomalies in log messages using TF-IDF + DBSCAN.
    """
    try:
        if len(log_messages) < 2:
            return {
                'n_anomalies': 0,
                'anomaly_percentage': 0.0,
                'is_anomaly': [False] * len(log_messages),
                'method': 'TF-IDF + DBSCAN'
            }

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )

        tfidf_matrix = vectorizer.fit_transform(log_messages)

        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        clusters = dbscan.fit_predict(tfidf_matrix)

        # Anomalies are noise points (-1)
        is_anomaly = clusters == -1
        n_anomalies = int(np.sum(is_anomaly))

        return {
            'n_anomalies': n_anomalies,
            'anomaly_percentage': (n_anomalies / len(log_messages)) * 100,
            'is_anomaly': is_anomaly.tolist(),
            'clusters': clusters.tolist(),
            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
            'method': 'TF-IDF + DBSCAN'
        }

    except Exception as e:
        logger.error(f"Error in TF-IDF anomaly detection: {e}", exc_info=True)
        raise


# ==================== NUMERIC OUTLIER DETECTION ====================

def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Detect outliers using Interquartile Range (IQR) method.
    Fast, interpretable, no training needed.

    Args:
        df: Input dataframe
        columns: Columns to analyze (default: all numeric)

    Returns:
        Dictionary with outlier flags, scores, and summary
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns found for outlier detection")

    outlier_flags = pd.DataFrame(index=df.index)
    outlier_scores = pd.DataFrame(index=df.index)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Flag outliers
        outlier_flags[col] = (df[col] < lower_bound) | (df[col] > upper_bound)

        # Score: distance from bounds normalized
        outlier_scores[col] = np.where(
            df[col] < lower_bound,
            (lower_bound - df[col]) / (IQR + 1e-10),
            np.where(df[col] > upper_bound, (df[col] - upper_bound) / (IQR + 1e-10), 0)
        )

    # Aggregate: any column flagged = outlier
    is_outlier = outlier_flags.any(axis=1)
    outlier_score = outlier_scores.max(axis=1)

    n_outliers = is_outlier.sum()

    return {
        'is_outlier': is_outlier.tolist(),
        'outlier_score': outlier_score.tolist(),
        'n_outliers': int(n_outliers),
        'outlier_percentage': float(n_outliers / len(df) * 100),
        'method': 'IQR',
        'columns_analyzed': columns
    }


def detect_outliers_zscore(df: pd.DataFrame, columns: Optional[List[str]] = None,
                           threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detect outliers using Z-score method.
    Fast, assumes normal distribution.

    Args:
        df: Input dataframe
        columns: Columns to analyze (default: all numeric)
        threshold: Z-score threshold (default: 3.0)

    Returns:
        Dictionary with outlier flags, scores, and summary
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns found for outlier detection")

    z_scores = np.abs(stats.zscore(df[columns].fillna(df[columns].median()), nan_policy='omit'))

    is_outlier = (z_scores > threshold).any(axis=1)
    outlier_score = z_scores.max(axis=1)

    n_outliers = is_outlier.sum()

    return {
        'is_outlier': is_outlier.tolist(),
        'outlier_score': outlier_score.tolist(),
        'n_outliers': int(n_outliers),
        'outlier_percentage': float(n_outliers / len(df) * 100),
        'method': 'Z-Score',
        'threshold': threshold,
        'columns_analyzed': columns
    }


@track_time
def detect_anomalies_iforest(df: pd.DataFrame, columns: Optional[List[str]] = None,
                             contamination: float = 0.1) -> Dict[str, Any]:
    """
    Detect anomalies using Isolation Forest.
    Best for high-dimensional data, handles non-linear patterns.

    Args:
        df: Input dataframe
        columns: Columns to analyze (default: all numeric)
        contamination: Expected proportion of anomalies (default: 0.1)

    Returns:
        Dictionary with anomaly flags, scores, and summary
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if not columns:
        raise ValueError("No numeric columns found for anomaly detection")

    X = df[columns].fillna(df[columns].median())

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
        max_samples='auto'
    )
    predictions = model.fit_predict(X)
    scores = model.score_samples(X)

    is_anomaly = predictions == -1
    anomaly_score = -scores  # Invert so higher = more anomalous

    n_anomalies = is_anomaly.sum()

    return {
        'is_anomaly': is_anomaly.tolist(),
        'anomaly_score': anomaly_score.tolist(),
        'n_anomalies': int(n_anomalies),
        'anomaly_percentage': float(n_anomalies / len(df) * 100),
        'method': 'Isolation Forest',
        'contamination': contamination,
        'columns_analyzed': columns
    }


@track_time
def linear_regression(df: pd.DataFrame, target_col: str,
                     feature_cols: Optional[List[str]] = None,
                     test_size: float = 0.2,
                     model_name: Optional[str] = None,
                     incremental: bool = True) -> Dict[str, Any]:
    """
    Perform linear regression with Ridge regularization and incremental learning support.

    Args:
        df: Input dataframe
        target_col: Target column name
        feature_cols: Feature columns (default: all numeric except target)
        test_size: Fraction for test set (default: 0.2)
        model_name: Name to save model (optional)
        incremental: If True, load existing model and continue training (default: True)

    Returns:
        Dictionary with predictions, metrics, and feature importance
    """
    if feature_cols is None:
        feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                       if col != target_col]

    if not feature_cols:
        raise ValueError("No feature columns found for regression")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    # Remove rows with missing target
    df_clean = df.dropna(subset=[target_col])

    # Fill missing features with median
    X = df_clean[feature_cols].fillna(df_clean[feature_cols].median())
    y = df_clean[target_col]

    # Split train/test
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Try to load existing model for incremental learning
    existing_model = None
    existing_scaler = None
    if incremental and model_name:
        result = model_manager.get_regression_model(model_name)
        if result:
            existing_model, existing_scaler = result
            logger.info(f"Loaded existing model '{model_name}' for incremental training")

    # Scale features
    if existing_scaler:
        scaler = existing_scaler
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Train model - use SGD for incremental learning or Ridge for batch
    if existing_model and isinstance(existing_model, SGDRegressor):
        # Continue training existing SGD model
        model = existing_model
        model.partial_fit(X_train_scaled, y_train)
        logger.info(f"Incremental training on {len(X_train)} samples")
    elif incremental and model_name:
        # Create new SGD model for incremental learning
        model = SGDRegressor(
            loss='squared_error',
            penalty='l2',
            alpha=0.0001,
            max_iter=1000,
            tol=1e-3,
            random_state=42,
            warm_start=True
        )
        model.fit(X_train_scaled, y_train)
        logger.info(f"Created new SGD model for incremental learning")
    else:
        # Batch training with Ridge
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        logger.info(f"Batch training with Ridge regression")

    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    # Metrics
    metrics = {
        'train_r2': float(r2_score(y_train, y_pred_train)),
        'test_r2': float(r2_score(y_test, y_pred_test)),
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
        'test_mae': float(mean_absolute_error(y_test, y_pred_test))
    }

    # Feature importance
    feature_importance = [
        {'feature': feat, 'coefficient': float(coef)}
        for feat, coef in sorted(
            zip(feature_cols, model.coef_),
            key=lambda x: abs(x[1]),
            reverse=True
        )
    ]

    # Save model if name provided
    if model_name:
        model_manager.save_regression_model(model_name, model, scaler)
        # Persist to disk immediately
        save_models()
        logger.info(f"Model '{model_name}' saved to disk")

    # Predictions for all data
    all_predictions = np.concatenate([y_pred_train, y_pred_test])

    return {
        'predictions': all_predictions.tolist(),
        'metrics': metrics,
        'feature_importance': feature_importance,
        'model_saved': model_name is not None,
        'model_name': model_name,
        'incremental': incremental
    }


@track_time
def forecast_exponential_smoothing(df: pd.DataFrame, value_col: str,
                                  time_col: Optional[str] = None,
                                  horizon: int = 10,
                                  alpha: float = 0.3) -> Dict[str, Any]:
    """
    Simple exponential smoothing forecast.
    Fast, works well for data without strong trend/seasonality.

    Args:
        df: Input dataframe
        value_col: Column to forecast
        time_col: Time column (optional, uses index if None)
        horizon: Number of periods to forecast
        alpha: Smoothing parameter (0-1, higher = more weight on recent)

    Returns:
        Dictionary with forecasts and metrics
    """
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")

    if time_col and time_col in df.columns:
        df = df.sort_values(time_col)

    values = df[value_col].dropna().values

    if len(values) < 2:
        raise ValueError("Need at least 2 data points for forecasting")

    # Simple exponential smoothing
    smoothed = [values[0]]
    for i in range(1, len(values)):
        smoothed.append(alpha * values[i] + (1 - alpha) * smoothed[-1])

    # Forecast (flat forecast at last smoothed value)
    last_smoothed = smoothed[-1]
    forecast = [last_smoothed] * horizon

    # Calculate error metrics on historical data
    errors = values[1:] - smoothed[:-1]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))

    return {
        'forecast': [float(x) for x in forecast],
        'smoothed_values': [float(x) for x in smoothed],
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'alpha': alpha
        },
        'horizon': int(horizon),
        'method': 'Exponential Smoothing'
    }


@track_time
def forecast_prophet(df: pd.DataFrame, value_col: str,
                    time_col: Optional[str] = None,
                    horizon: int = 10) -> Dict[str, Any]:
    """
    Forecast using Facebook Prophet (best for time-series with seasonality).

    Args:
        df: Input dataframe
        value_col: Column to forecast
        time_col: Time column (optional, uses index if None)
        horizon: Number of periods to forecast

    Returns:
        Dictionary with forecasts and metrics
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet not installed. Install with: pip install prophet")

    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")

    try:
        # Prepare data for Prophet (requires 'ds' and 'y' columns)
        prophet_df = pd.DataFrame()

        if time_col and time_col in df.columns:
            prophet_df['ds'] = pd.to_datetime(df[time_col])
        else:
            # Create date range if no time column
            prophet_df['ds'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')

        prophet_df['y'] = df[value_col].values

        # Remove any NaN values
        prophet_df = prophet_df.dropna()

        if len(prophet_df) < 2:
            raise ValueError("Need at least 2 data points for Prophet forecasting")

        # Train Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True if len(prophet_df) > 14 else False,
            yearly_seasonality=False,
            seasonality_mode='multiplicative'
        )

        # Suppress Prophet's verbose output
        import logging as prophet_logging
        prophet_logging.getLogger('prophet').setLevel(prophet_logging.WARNING)

        model.fit(prophet_df)

        # Make future dataframe
        future = model.make_future_dataframe(periods=horizon)
        forecast_result = model.predict(future)

        # Extract forecast values
        forecast_values = forecast_result['yhat'].tail(horizon).values

        # Calculate metrics on historical data
        historical_pred = forecast_result['yhat'].head(len(prophet_df)).values
        residuals = prophet_df['y'].values - historical_pred
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(residuals**2)))

        return {
            'forecast': forecast_values.tolist(),
            'forecast_lower': forecast_result['yhat_lower'].tail(horizon).values.tolist(),
            'forecast_upper': forecast_result['yhat_upper'].tail(horizon).values.tolist(),
            'metrics': {
                'mae': mae,
                'rmse': rmse
            },
            'horizon': horizon,
            'method': 'Prophet'
        }

    except Exception as e:
        logger.error(f"Error in Prophet forecasting: {e}", exc_info=True)
        raise


@track_time
def forecast_linear_trend(df: pd.DataFrame, value_col: str,
                         time_col: Optional[str] = None,
                         horizon: int = 10) -> Dict[str, Any]:
    """
    Linear trend forecasting.
    Simple, interpretable, works well for data with linear trend.

    Args:
        df: Input dataframe
        value_col: Column to forecast
        time_col: Time column (optional, uses index if None)
        horizon: Number of periods to forecast

    Returns:
        Dictionary with forecasts and metrics
    """
    if value_col not in df.columns:
        raise ValueError(f"Value column '{value_col}' not found in dataframe")

    if time_col and time_col in df.columns:
        df = df.sort_values(time_col)

    values = df[value_col].dropna().values

    if len(values) < 2:
        raise ValueError("Need at least 2 data points for forecasting")

    X = np.arange(len(values)).reshape(-1, 1)

    # Fit linear trend
    model = Ridge(alpha=0.1)  # Small regularization
    model.fit(X, values)

    # Historical fit
    fitted = model.predict(X)

    # Forecast
    X_future = np.arange(len(values), len(values) + horizon).reshape(-1, 1)
    forecast = model.predict(X_future)

    # Metrics
    residuals = values - fitted
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals**2)))

    return {
        'forecast': forecast.tolist(),
        'fitted_values': fitted.tolist(),
        'metrics': {
            'mae': mae,
            'rmse': rmse,
            'slope': float(model.coef_[0]),
            'intercept': float(model.intercept_)
        },
        'horizon': horizon,
        'method': 'Linear Trend'
    }


@track_time
def generate_visualizations(anomaly_scores: np.ndarray, patterns: np.ndarray, timestamps: pd.Series) -> str:
    """
    Generate visualization of log analysis results.

    Args:
        anomaly_scores: Anomaly scores for each log
        patterns: Cluster labels for each log
        timestamps: Timestamps for each log

    Returns:
        Base64-encoded PNG image
    """
    try:
        plt.figure(figsize=(12, 6), dpi=config.VISUALIZATION_DPI)
        scatter = plt.scatter(timestamps, anomaly_scores, c=patterns, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Pattern Cluster')
        plt.title('Log Analysis: Anomaly Scores Over Time')
        plt.xlabel('Timestamp')
        plt.ylabel('Anomaly Score')
        plt.xticks(rotation=45)
        plt.tight_layout()

        img = io.BytesIO()
        plt.savefig(img, format=config.VISUALIZATION_FORMAT, bbox_inches='tight')
        plt.close()
        img.seek(0)

        encoded = base64.b64encode(img.getvalue()).decode('utf-8')
        logger.debug("Generated visualization successfully")
        return encoded

    except Exception as e:
        logger.error(f"Error generating visualization: {e}", exc_info=True)
        # Return empty image on error
        return ""


# API Endpoints

@app.before_request
def before_request():
    """Track request start time."""
    g.start_time = time.time()


@app.after_request
def after_request(response):
    """Log request completion and add headers."""
    if hasattr(g, 'start_time'):
        elapsed = time.time() - g.start_time
        logger.info(f"{request.method} {request.path} - {response.status_code} - {elapsed:.3f}s")

    # Add security headers
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'

    return response


@app.errorhandler(400)
def bad_request(error):
    """Handle bad request errors."""
    logger.warning(f"Bad request: {error}")
    return jsonify({'error': 'Bad request', 'message': str(error)}), 400


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle request too large errors."""
    logger.warning(f"Request too large: {error}")
    return jsonify({'error': 'Request too large', 'max_size': config.MAX_CONTENT_LENGTH}), 413


@app.errorhandler(429)
def ratelimit_handler(error):
    """Handle rate limit errors."""
    logger.warning(f"Rate limit exceeded: {error}")
    return jsonify({'error': 'Rate limit exceeded', 'message': str(error)}), 429


@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors."""
    logger.error(f"Internal server error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        datasets = data_store.list_datasets()
        models = model_manager.list_models()

        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': config.MODEL_VERSION,
            'datasets_loaded': len(datasets),
            'models_saved': len(models)
        }

        return jsonify(health_status), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """Get detailed system status."""
    try:
        datasets = data_store.list_datasets()
        models = model_manager.list_models()

        dataset_info = []
        for name in datasets:
            meta = data_store.get_metadata(name)
            if meta:
                dataset_info.append({
                    'name': meta.name,
                    'rows': meta.shape[0],
                    'columns': meta.shape[1],
                    'numeric_cols': len(meta.numeric_cols),
                    'categorical_cols': len(meta.categorical_cols),
                    'datetime_cols': len(meta.datetime_cols)
                })

        status_info = {
            'timestamp': datetime.now().isoformat(),
            'version': config.MODEL_VERSION,
            'datasets': dataset_info,
            'models': models,
            'configuration': {
                'max_batch_size': config.MAX_BATCH_SIZE,
                'rate_limit_enabled': config.RATE_LIMIT_ENABLED
            }
        }

        return jsonify(status_info), 200

    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/upload_data', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def upload_data():
    """
    Upload tabular data (CSV, JSON, or dict format).

    Expected JSON body:
    {
        "name": "my_dataset",
        "data": [...] or {"col1": [...], "col2": [...]},
        "format": "records" | "dict" (optional, default: auto-detect)
    }

    Returns dataset info and summary statistics.
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        name = request.json.get('name')
        data = request.json.get('data')
        data_format = request.json.get('format', 'auto')

        if not name:
            return jsonify({'error': 'Dataset name is required'}), 400

        if not data:
            return jsonify({'error': 'Data is required'}), 400

        logger.info(f"Uploading dataset '{name}'")

        # Convert to DataFrame
        try:
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                return jsonify({'error': 'Data must be a list or dict'}), 400
        except Exception as e:
            return jsonify({'error': f'Failed to parse data: {str(e)}'}), 400

        if df.empty:
            return jsonify({'error': 'Dataset is empty'}), 400

        if len(df) > config.MAX_BATCH_SIZE:
            return jsonify({
                'error': f'Dataset size ({len(df)}) exceeds maximum ({config.MAX_BATCH_SIZE})'
            }), 400

        # Store dataset
        info = data_store.add_dataset(name, df)

        # Generate summary statistics
        summary = {
            'name': info.name,
            'shape': {'rows': info.shape[0], 'columns': info.shape[1]},
            'columns': info.columns,
            'dtypes': info.dtypes,
            'numeric_columns': info.numeric_cols,
            'categorical_columns': info.categorical_cols,
            'datetime_columns': info.datetime_cols,
            'missing_values': df.isnull().sum().to_dict(),
            'sample_data': df.head(5).to_dict('records')
        }

        logger.info(f"Dataset '{name}' uploaded successfully: {info.shape}")
        return jsonify(summary), 200

    except Exception as e:
        logger.error(f"Error uploading data: {e}", exc_info=True)
        return jsonify({'error': 'Upload failed', 'details': str(e)}), 500


@app.route('/upload_file', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def upload_file():
    """
    SMART FILE UPLOAD - Turnkey solution for any file type.

    Automatically handles:
    - Raw log files (.log, .txt) → parsed into structured data
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - JSON files (.json)

    Body: {"file": "path/to/file", "name": "dataset_name" (optional)}

    Returns: Dataset info + capabilities (text anomaly detection, forecasting, etc.)
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        file_path = request.json.get('file')
        dataset_name = request.json.get('name')

        if not file_path:
            return jsonify({'error': 'file path is required'}), 400

        if not dataset_name:
            dataset_name = os.path.basename(file_path).rsplit('.', 1)[0]

        logger.info(f"Smart upload: {file_path}")

        # Auto-detect and load based on file extension
        if file_path.endswith(('.log', '.txt')):
            df = parse_log_file(file_path)
            file_type = 'log'
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            file_type = 'csv'
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
            file_type = 'excel'
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
            file_type = 'json'
        else:
            return jsonify({'error': 'Unsupported file. Use: .log, .txt, .csv, .xlsx, .json'}), 400

        # Store dataset
        info = data_store.add_dataset(dataset_name, df)

        # Detect capabilities
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Check for time columns
        time_columns = []
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                time_columns.append(col)

        response = {
            'status': 'success',
            'dataset': dataset_name,
            'file_type': file_type,
            'rows': info.shape[0],
            'columns': info.shape[1],
            'numeric_columns': numeric_columns,
            'text_columns': text_columns,
            'time_columns': time_columns,
            'capabilities': {
                'outlier_detection': len(numeric_columns) > 0,
                'text_anomaly_detection': len(text_columns) > 0,
                'forecasting': len(numeric_columns) > 0,
                'regression': len(numeric_columns) > 1
            }
        }

        logger.info(f"✓ Smart upload complete: {file_type} → {dataset_name} ({len(df)} rows)")
        return jsonify(response), 200

    except FileNotFoundError:
        return jsonify({'error': f'File not found: {file_path}'}), 404
    except Exception as e:
        logger.error(f"Error in smart upload: {e}", exc_info=True)
        return jsonify({'error': 'Upload failed', 'details': str(e)}), 500


@app.route('/detect_log_anomalies', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def detect_log_anomalies():
    """
    Detect anomalies in log messages using TF-IDF + DBSCAN.

    Body: {
        "dataset": "log_dataset",
        "text_column": "raw_message" (default: auto-detect),
        "eps": 0.5,
        "min_samples": 2
    }
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        text_column = request.json.get('text_column')
        eps = request.json.get('eps', 0.5)
        min_samples = request.json.get('min_samples', 2)

        if not dataset_name:
            return jsonify({'error': 'dataset is required'}), 400

        df = data_store.get_dataset(dataset_name)
        if df is None:
            return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404

        # Auto-detect text column if not specified
        if not text_column:
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if not text_cols:
                return jsonify({'error': 'No text columns found for anomaly detection'}), 400
            text_column = text_cols[0]

        if text_column not in df.columns:
            return jsonify({'error': f'Column "{text_column}" not found'}), 404

        logger.info(f"Detecting log anomalies in '{dataset_name}' column '{text_column}'")

        # Get log messages
        log_messages = df[text_column].astype(str).tolist()

        # Detect anomalies
        result = detect_log_anomalies_tfidf(log_messages, eps, min_samples)

        # Add sample anomalies
        anomaly_indices = [i for i, is_anom in enumerate(result['is_anomaly']) if is_anom]
        result['sample_anomalies'] = [
            {'index': idx, 'message': log_messages[idx][:200]}
            for idx in anomaly_indices[:10]
        ]

        logger.info(f"✓ Found {result['n_anomalies']} log anomalies")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error detecting log anomalies: {e}", exc_info=True)
        return jsonify({'error': 'Log anomaly detection failed', 'details': str(e)}), 500


@app.route('/detect_outliers', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def detect_outliers_endpoint():
    """
    Detect outliers in a dataset.

    Expected JSON body:
    {
        "dataset": "my_dataset",
        "method": "iqr" | "zscore" | "iforest",
        "columns": ["col1", "col2"] (optional),
        "threshold": 3.0 (for zscore),
        "contamination": 0.1 (for iforest)
    }
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        method = request.json.get('method', 'iqr')
        columns = request.json.get('columns')
        threshold = request.json.get('threshold', 3.0)
        contamination = request.json.get('contamination', 0.1)

        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'}), 400

        df = data_store.get_dataset(dataset_name)
        if df is None:
            return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404

        logger.info(f"Detecting outliers in '{dataset_name}' using {method}")

        # Detect outliers based on method
        if method == 'iqr':
            result = detect_outliers_iqr(df, columns)
        elif method == 'zscore':
            result = detect_outliers_zscore(df, columns, threshold)
        elif method == 'iforest':
            result = detect_anomalies_iforest(df, columns, contamination)
            # Normalize key names for consistency
            result['is_outlier'] = result.pop('is_anomaly')
            result['outlier_score'] = result.pop('anomaly_score')
            result['n_outliers'] = result.pop('n_anomalies')
            result['outlier_percentage'] = result.pop('anomaly_percentage')
        else:
            return jsonify({'error': f'Unknown method: {method}. Use iqr, zscore, or iforest'}), 400

        # Add outlier indices
        outlier_indices = [i for i, is_out in enumerate(result['is_outlier']) if is_out]
        result['outlier_indices'] = outlier_indices[:100]  # Limit to first 100

        logger.info(f"Outlier detection complete: {result.get('n_outliers', 0)} outliers found")
        return jsonify(result), 200

    except ValueError as e:
        logger.warning(f"Validation error in detect_outliers: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Error detecting outliers: {e}", exc_info=True)
        return jsonify({'error': 'Outlier detection failed', 'details': str(e)}), 500


@app.route('/regression', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def regression_endpoint():
    """
    Perform linear regression.

    Expected JSON body:
    {
        "dataset": "my_dataset",
        "target": "target_column",
        "features": ["feature1", "feature2"] (optional),
        "test_size": 0.2 (optional),
        "model_name": "my_model" (optional, to save)
    }
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        target_col = request.json.get('target')
        feature_cols = request.json.get('features')
        test_size = request.json.get('test_size', 0.2)
        model_name = request.json.get('model_name')

        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'}), 400

        if not target_col:
            return jsonify({'error': 'Target column is required'}), 400

        df = data_store.get_dataset(dataset_name)
        if df is None:
            return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404

        logger.info(f"Running regression on '{dataset_name}', target='{target_col}'")

        result = linear_regression(df, target_col, feature_cols, test_size, model_name)

        logger.info(f"Regression complete: R²={result['metrics']['test_r2']:.3f}")
        return jsonify(result), 200

    except ValueError as e:
        logger.warning(f"Validation error in regression: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Error in regression: {e}", exc_info=True)
        return jsonify({'error': 'Regression failed', 'details': str(e)}), 500


@app.route('/forecast', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def forecast_endpoint():
    """
    Forecast time series data using Prophet (default) or simple methods.

    Expected JSON body:
    {
        "dataset": "my_dataset",
        "value_column": "value",
        "time_column": "timestamp" (optional),
        "method": "prophet" | "exponential" | "linear" (default: prophet),
        "horizon": 10,
        "alpha": 0.3 (for exponential smoothing)
    }
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        value_col = request.json.get('value_column')
        time_col = request.json.get('time_column')
        method = request.json.get('method', 'prophet' if PROPHET_AVAILABLE else 'linear')
        horizon = request.json.get('horizon', 10)
        alpha = request.json.get('alpha', 0.3)

        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'}), 400

        if not value_col:
            return jsonify({'error': 'Value column is required'}), 400

        df = data_store.get_dataset(dataset_name)
        if df is None:
            return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404

        logger.info(f"Forecasting '{value_col}' in '{dataset_name}' using {method}")

        if method == 'prophet':
            if not PROPHET_AVAILABLE:
                return jsonify({'error': 'Prophet not installed. Use method: linear or exponential'}), 400
            result = forecast_prophet(df, value_col, time_col, horizon)
        elif method == 'exponential':
            result = forecast_exponential_smoothing(df, value_col, time_col, horizon, alpha)
        elif method == 'linear':
            result = forecast_linear_trend(df, value_col, time_col, horizon)
        else:
            return jsonify({'error': f'Unknown method: {method}. Use prophet, exponential, or linear'}), 400

        logger.info(f"Forecast complete: {horizon} periods, MAE={result['metrics']['mae']:.3f}")
        return jsonify(result), 200

    except ValueError as e:
        logger.warning(f"Validation error in forecast: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Error in forecast: {e}", exc_info=True)
        return jsonify({'error': 'Forecast failed', 'details': str(e)}), 500


@app.route('/delete_dataset', methods=['POST'])
@limiter.limit("10/minute")
def delete_dataset():
    """Delete a dataset from memory."""
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'}), 400

        success = data_store.delete_dataset(dataset_name)
        if success:
            return jsonify({'status': 'success', 'message': f'Dataset "{dataset_name}" deleted'}), 200
        else:
            return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404

    except Exception as e:
        logger.error(f"Error deleting dataset: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/list_datasets', methods=['GET'])
def list_datasets():
    """List all stored datasets."""
    try:
        datasets = data_store.list_datasets()
        dataset_info = []

        for name in datasets:
            meta = data_store.get_metadata(name)
            if meta:
                dataset_info.append({
                    'name': meta.name,
                    'rows': meta.shape[0],
                    'columns': meta.shape[1],
                    'numeric_cols': len(meta.numeric_cols),
                    'uploaded_at': meta.timestamp.isoformat()
                })

        return jsonify({'datasets': dataset_info}), 200

    except Exception as e:
        logger.error(f"Error listing datasets: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/train_model', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def train_model():
    """
    Train THE UNIFIED MODEL on a dataset (incremental learning).
    All datasets contribute to a single, continuously growing model.

    Body: {"dataset": "name", "target": "col", "features": [...] (optional)}

    Note: model_name is always "unified_model" - all data trains one model.
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        target_col = request.json.get('target')
        feature_cols = request.json.get('features')
        test_size = request.json.get('test_size', 0.2)

        if not all([dataset_name, target_col]):
            return jsonify({'error': 'dataset and target are required'}), 400

        df = data_store.get_dataset(dataset_name)
        if df is None:
            return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404

        logger.info(f"Training UNIFIED MODEL on '{dataset_name}' ({len(df)} samples)")

        # Always use unified_model with incremental learning
        result = linear_regression(
            df,
            target_col,
            feature_cols,
            test_size,
            model_name="unified_model",
            incremental=True
        )

        result['model_name'] = 'unified_model'
        result['dataset_source'] = dataset_name
        result['samples_added'] = len(df)

        logger.info(f"✓ Unified model trained on {len(df)} samples from '{dataset_name}'")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error training model: {e}", exc_info=True)
        return jsonify({'error': 'Training failed', 'details': str(e)}), 500


@app.route('/predict', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def predict():
    """
    Use a trained model to make predictions.

    Body: {"model_name": "name", "dataset": "name"} OR {"model_name": "name", "data": [...]}
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        model_name = request.json.get('model_name')
        dataset_name = request.json.get('dataset')
        data = request.json.get('data')

        if not model_name:
            return jsonify({'error': 'model_name is required'}), 400

        result = model_manager.get_regression_model(model_name)
        if result is None:
            return jsonify({'error': f'Model "{model_name}" not found'}), 404

        model, scaler = result

        # Get data
        if dataset_name:
            df = data_store.get_dataset(dataset_name)
            if df is None:
                return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404
        elif data:
            df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame(data)
        else:
            return jsonify({'error': 'Either dataset or data is required'}), 400

        # Prepare features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].fillna(df[numeric_cols].median())

        # Scale and predict
        X_scaled = scaler.transform(X) if scaler else X.values
        predictions = model.predict(X_scaled)

        return jsonify({'predictions': predictions.tolist(), 'n_predictions': len(predictions)}), 200

    except Exception as e:
        logger.error(f"Error making predictions: {e}", exc_info=True)
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@app.route('/list_models', methods=['GET'])
def list_models():
    """List all trained models."""
    try:
        models = model_manager.list_models()
        return jsonify({'models': models}), 200
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ==================== SIMPLE WORKFLOW ENDPOINTS ====================

@app.route('/train', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def train():
    """
    Simple train endpoint - load data and train THE UNIFIED MODEL.
    All data trains a single, continuously growing model.

    Body: {"file": "path/to/data.json|csv|xlsx", "target": "column_name" (optional)}
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        file_path = request.json.get('file')
        target_col = request.json.get('target')

        if not file_path:
            return jsonify({'error': 'file path is required'}), 400

        logger.info(f"Training UNIFIED MODEL on file: {file_path}")

        # Load data from file based on extension
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            return jsonify({'error': 'Unsupported file format. Use .csv, .xlsx, or .json'}), 400

        dataset_name = os.path.basename(file_path).rsplit('.', 1)[0]

        # Store dataset
        info = data_store.add_dataset(dataset_name, df)

        # Train THE UNIFIED MODEL if target specified
        model_trained = False
        model_metrics = None
        total_samples = 0

        if target_col and target_col in df.columns:
            # ALWAYS use the same model name - "unified_model"
            result = linear_regression(
                df,
                target_col,
                model_name="unified_model",  # Single unified model
                incremental=True  # Always incremental
            )
            model_trained = True
            model_metrics = result['metrics']
            total_samples = len(df)

            logger.info(f"✓ Unified model trained on {total_samples} new samples from {dataset_name}")

        response = {
            'status': 'success',
            'dataset': dataset_name,
            'rows': info.shape[0],
            'columns': info.shape[1],
            'numeric_columns': info.numeric_cols,
            'model_trained': model_trained,
            'model_name': 'unified_model' if model_trained else None,
            'samples_added': total_samples,
            'incremental': True
        }

        if model_metrics:
            response['model_metrics'] = model_metrics

        logger.info(f"Training complete: {dataset_name} → unified_model")
        return jsonify(response), 200

    except FileNotFoundError:
        return jsonify({'error': f'File not found: {file_path}'}), 404
    except Exception as e:
        logger.error(f"Error in train: {e}", exc_info=True)
        return jsonify({'error': 'Training failed', 'details': str(e)}), 500


@app.route('/analyze', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def analyze():
    """
    Comprehensive analysis - runs all analytics and returns visualization.

    Body: {"dataset": "name" OR "file": "path.json", "target": "col" (optional)}
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        file_path = request.json.get('file')
        target_col = request.json.get('target')

        # Load data
        if file_path:
            # Load based on file extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return jsonify({'error': 'Unsupported file format. Use .csv, .xlsx, or .json'}), 400

            dataset_name = os.path.basename(file_path).rsplit('.', 1)[0]
            data_store.add_dataset(dataset_name, df)
        elif dataset_name:
            df = data_store.get_dataset(dataset_name)
            if df is None:
                return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404
        else:
            return jsonify({'error': 'Either "dataset" or "file" is required'}), 400

        logger.info(f"Analyzing dataset: {dataset_name}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            return jsonify({'error': 'No numeric columns found'}), 400

        # Run all analytics
        outliers_iqr = detect_outliers_iqr(df, numeric_cols)
        anomalies = detect_anomalies_iforest(df, numeric_cols, contamination=0.1)

        # Regression if target specified
        regression_results = None
        if target_col and target_col in df.columns:
            regression_results = linear_regression(df, target_col, test_size=0.2)

        # Forecasting on first numeric column
        forecast_col = numeric_cols[0]

        # Use Prophet if available, otherwise fallback to simple methods
        if PROPHET_AVAILABLE:
            try:
                forecast_main = forecast_prophet(df, forecast_col, horizon=5)
                forecast_method = 'prophet'
            except Exception as e:
                logger.warning(f"Prophet forecasting failed, using linear trend: {e}")
                forecast_main = forecast_linear_trend(df, forecast_col, horizon=5)
                forecast_method = 'linear'
        else:
            forecast_main = forecast_linear_trend(df, forecast_col, horizon=5)
            forecast_method = 'linear'

        # Also include simple forecast for comparison
        forecast_exp = forecast_exponential_smoothing(df, forecast_col, horizon=5, alpha=0.3)

        # Generate visualization
        viz, viz_path = generate_analysis_visualization(df, numeric_cols, outliers_iqr, anomalies, dataset_name)

        results = {
            'dataset': dataset_name,
            'summary': {
                'rows': len(df),
                'columns': len(df.columns),
                'numeric_columns': len(numeric_cols)
            },
            'outliers': {
                'method': 'IQR',
                'count': outliers_iqr['n_outliers'],
                'percentage': outliers_iqr['outlier_percentage'],
                'indices': [i for i, flag in enumerate(outliers_iqr['is_outlier']) if flag][:10]
            },
            'anomalies': {
                'method': 'Isolation Forest',
                'count': anomalies['n_anomalies'],
                'percentage': anomalies['anomaly_percentage']
            },
            'forecasts': {
                'primary': {
                    'method': forecast_method,
                    'values': forecast_main['forecast'],
                    'mae': forecast_main['metrics']['mae']
                },
                'exponential_smoothing': {
                    'values': forecast_exp['forecast'],
                    'mae': forecast_exp['metrics']['mae']
                }
            },
            'visualization': viz,
            'visualization_path': viz_path
        }

        if regression_results:
            results['regression'] = {
                'target': target_col,
                'r2': regression_results['metrics']['test_r2'],
                'rmse': regression_results['metrics']['test_rmse'],
                'top_features': regression_results['feature_importance'][:5]
            }

        logger.info(f"Analysis complete for {dataset_name}")
        return jsonify(results), 200

    except FileNotFoundError:
        return jsonify({'error': f'File not found: {file_path}'}), 404
    except Exception as e:
        logger.error(f"Error in analyze: {e}", exc_info=True)
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500


def generate_analysis_visualization(df: pd.DataFrame, numeric_cols: List[str],
                                    outliers: Dict, anomalies: Dict, dataset_name: str = 'data') -> Tuple[str, str]:
    """
    Generate comprehensive analysis visualization and save to analysis folder.

    Returns:
        Tuple of (base64_encoded_image, saved_file_path)
    """
    try:
        # Create analysis directory
        analysis_dir = 'analysis'
        os.makedirs(analysis_dir, exist_ok=True)

        # Generate timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{dataset_name}_analysis_{timestamp}.png"
        filepath = os.path.join(analysis_dir, filename)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=config.VISUALIZATION_DPI)
        fig.suptitle(f'Comprehensive Data Analysis - {dataset_name}', fontsize=16, fontweight='bold')

        # Determine best time axis for plotting
        time_col = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                time_col = col
                break
        if time_col is None:
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ('time', 'date')):
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    if parsed.notna().sum() >= len(df) * 0.5:
                        time_col = col
                        df = df.copy()
                        df[col] = parsed
                        break
        time_values = df[time_col] if time_col else pd.RangeIndex(len(df))

        # Prepare standardized values for fair comparison across metrics
        def _zscore(series: pd.Series) -> pd.Series:
            std = series.std(skipna=True)
            return (series - series.mean()) / (std if std and std != 0 else 1.0)

        standardized = df[numeric_cols[:3]].apply(_zscore)

        # Plot 1: Standardized data distribution
        ax1 = axes[0, 0]
        for col in standardized.columns:
            ax1.plot(time_values, standardized[col], label=col, alpha=0.75)
        ax1.set_title('Standardized Value Trends')
        ax1.set_xlabel(time_col or 'Index')
        ax1.set_ylabel('Z-score')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Choose primary column with highest variability for scatter plots
        primary_col = max(numeric_cols, key=lambda c: df[c].std(skipna=True) if df[c].std(skipna=True) is not None else 0)
        primary_series = df[primary_col]

        # Plot 2: Outliers referencing the primary column
        ax2 = axes[0, 1]
        outlier_flags = outliers['is_outlier']
        ax2.scatter(time_values, primary_series,
                    c=['#D7263D' if flag else '#6EA4BF' for flag in outlier_flags],
                    alpha=0.7, s=35, edgecolors='k', linewidths=0.2)
        ax2.set_title(f'Outliers (IQR) on {primary_col}: {outliers["n_outliers"]} rows')
        ax2.set_xlabel(time_col or 'Index')
        ax2.set_ylabel(primary_col)
        ax2.grid(True, alpha=0.3)

        # Plot 3: Anomalies highlighting the same primary column
        ax3 = axes[1, 0]
        anomaly_flags = anomalies['is_anomaly']
        ax3.scatter(time_values, primary_series,
                    c=['#FF9F1C' if flag else '#2EC4B6' for flag in anomaly_flags],
                    alpha=0.7, s=35, edgecolors='k', linewidths=0.2)
        ax3.set_title(f'Anomalies (Isolation Forest) on {primary_col}: {anomalies["n_anomalies"]}')
        ax3.set_xlabel(time_col or 'Index')
        ax3.set_ylabel(primary_col)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        stats_lines = [
            'Dataset Snapshot',
            '────────────────────',
            f"Rows: {len(df):,}",
            f"Columns: {len(df.columns)}",
            f"Numeric Columns: {len(numeric_cols)}",
            f"Time Axis: {time_col or 'index'}",
            '',
            'Outliers',
            f" • Method: IQR",
            f" • Count: {outliers['n_outliers']} ({outliers['outlier_percentage']:.1f}%)",
            '',
            'Anomalies',
            f" • Method: Isolation Forest",
            f" • Count: {anomalies['n_anomalies']} ({anomalies['anomaly_percentage']:.1f}%)",
            '',
            'Top Metrics',
            f" • {', '.join(numeric_cols[:3])}"
        ]
        ax4.text(0.05, 0.95, '\n'.join(stats_lines), fontsize=11,
                 family='monospace', verticalalignment='top')

        plt.tight_layout()

        # Save to file
        plt.savefig(filepath, format=config.VISUALIZATION_FORMAT, bbox_inches='tight')
        logger.info(f"Saved visualization to {filepath}")

        # Also create base64 for API response
        img = io.BytesIO()
        plt.savefig(img, format=config.VISUALIZATION_FORMAT, bbox_inches='tight')
        plt.close()
        img.seek(0)

        encoded = base64.b64encode(img.getvalue()).decode('utf-8')
        return encoded, filepath

    except Exception as e:
        logger.error(f"Error generating visualization: {e}", exc_info=True)
        return "", ""


# Model persistence functions

def save_models() -> None:
    """Save all regression models to disk."""
    try:
        model_dir = config.MODEL_DIR
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created model directory: {model_dir}")

        models = model_manager.list_models()
        for model_name in models:
            result = model_manager.get_regression_model(model_name)
            if result:
                model, scaler = result
                model_path = os.path.join(model_dir, f'{model_name}_{config.MODEL_VERSION}.joblib')
                scaler_path = os.path.join(model_dir, f'{model_name}_scaler_{config.MODEL_VERSION}.joblib')

                joblib.dump(model, model_path)
                if scaler:
                    joblib.dump(scaler, scaler_path)

                logger.info(f"Saved model '{model_name}' to {model_path}")

        logger.info(f"Saved {len(models)} models successfully")

    except Exception as e:
        logger.error(f"Error saving models: {e}", exc_info=True)
        raise


def load_models() -> None:
    """Load regression models from disk."""
    try:
        model_dir = config.MODEL_DIR

        if not os.path.exists(model_dir):
            logger.info(f"Model directory {model_dir} does not exist, skipping model loading")
            return

        # Find all model files
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.joblib') and 'scaler' not in f]

        for model_file in model_files:
            model_name = model_file.replace(f'_{config.MODEL_VERSION}.joblib', '')
            model_path = os.path.join(model_dir, model_file)
            scaler_path = os.path.join(model_dir, f'{model_name}_scaler_{config.MODEL_VERSION}.joblib')

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

            model_manager.save_regression_model(model_name, model, scaler)
            logger.info(f"Loaded model '{model_name}' from {model_path}")

        logger.info("Model loading complete")

    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        # Don't raise - allow app to start with fresh models


def initialize() -> None:
    """Initialize the application."""
    try:
        logger.info("Initializing Analytics API")
        logger.info(f"Configuration: {config.__dict__}")

        # Load existing models
        load_models()

        logger.info("Initialization complete")

    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        raise


# Initialize on startup
with app.app_context():
    initialize()


def cli_analyze(file_path: str, target: str = None):
    """CLI: Analyze a dataset."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {file_path}")
    print('='*60)

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith(('.log', '.txt')):
        df = parse_log_file(file_path)
    else:
        print(f"❌ Unsupported file format")
        return

    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("❌ No numeric columns found")
        return

    print(f"✓ Numeric columns: {', '.join(numeric_cols)}")

    print(f"\n🔍 OUTLIER DETECTION (IQR)")
    outliers = detect_outliers_iqr(df, numeric_cols)
    print(f"  Found: {outliers['n_outliers']} ({outliers['outlier_percentage']:.1f}%)")

    print(f"\n⚠️  ANOMALY DETECTION (Isolation Forest)")
    anomalies = detect_anomalies_iforest(df, numeric_cols)
    print(f"  Found: {anomalies['n_anomalies']} ({anomalies['anomaly_percentage']:.1f}%)")

    if len(numeric_cols) > 0:
        forecast_col = target if target and target in numeric_cols else numeric_cols[0]
        print(f"\n📈 FORECASTING ({forecast_col})")
        forecast = forecast_linear_trend(df, forecast_col, horizon=5)
        print(f"  Method: Linear Trend")
        print(f"  Next 5: {[round(x, 1) for x in forecast['forecast']]}")
        print(f"  MAE: {forecast['metrics']['mae']:.2f}")

    if target and target in numeric_cols:
        print(f"\n📉 REGRESSION (target: {target})")
        result = linear_regression(df, target, test_size=0.2)
        print(f"  R²: {result['metrics']['test_r2']:.3f}")
        print(f"  RMSE: {result['metrics']['test_rmse']:.3f}")

    print(f"\n{'='*60}")
    print("✓ Analysis complete!")
    print('='*60)


def cli_train(file_path: str, target: str):
    """CLI: Train unified model."""
    print(f"\n{'='*60}")
    print(f"TRAINING UNIFIED MODEL: {file_path}")
    print('='*60)

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    else:
        print(f"❌ Unsupported file format")
        return

    print(f"✓ Loaded {len(df)} rows")

    if target not in df.columns:
        print(f"❌ Target column '{target}' not found")
        return

    result = linear_regression(df, target, model_name="unified_model", incremental=True)

    print(f"✓ Model: unified_model")
    print(f"✓ Samples added: {len(df)}")
    print(f"✓ R²: {result['metrics']['test_r2']:.3f}")
    print(f"✓ RMSE: {result['metrics']['test_rmse']:.3f}")

    print(f"\n{'='*60}")
    print("✓ Training complete! Model saved to models/")
    print('='*60)


def cli_main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analytics API - Turnkey ML Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                                    # Start server
  python app.py analyze data/sales_data.csv        # Analyze dataset
  python app.py train data/sales_data.csv sales    # Train model
  python app.py status                             # Show status
        """
    )

    parser.add_argument('command', nargs='?', default='serve',
                       choices=['serve', 'analyze', 'train', 'status'],
                       help='Command to run')
    parser.add_argument('file', nargs='?', help='File path')
    parser.add_argument('target', nargs='?', help='Target column for training')
    parser.add_argument('--port', type=int, default=5001, help='Server port')

    args = parser.parse_args()

    if args.command == 'analyze':
        if not args.file:
            print("❌ File path required")
            sys.exit(1)
        cli_analyze(args.file, args.target)

    elif args.command == 'train':
        if not args.file or not args.target:
            print("❌ File and target required")
            sys.exit(1)
        cli_train(args.file, args.target)

    elif args.command == 'status':
        print(f"\n{'='*60}")
        print("SYSTEM STATUS")
        print('='*60)
        print(f"Datasets: {len(data_store.datasets)}")

        # List model files
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib') and 'scaler' not in f] if os.path.exists('models') else []
        print(f"Models: {len(model_files)}")
        if model_files:
            print("\nSaved models:")
            for f in model_files:
                print(f"  • {f}")
        print('='*60)

    else:  # serve
        logger.warning("Running in development mode. Use gunicorn for production.")
        app.run(host='0.0.0.0', port=args.port, debug=config.DEBUG)


if __name__ == '__main__':
    cli_main()
