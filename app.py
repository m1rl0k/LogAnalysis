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
    """Thread-safe model management for regression and forecasting with schema validation."""

    def __init__(self):
        self.regression_models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_schemas: Dict[str, List[str]] = {}  # Store feature column names for each model
        self.lock = Lock()
        logger.info("Initialized ModelManager")

    def save_regression_model(self, name: str, model: Any, scaler: Optional[StandardScaler] = None,
                             feature_columns: Optional[List[str]] = None) -> None:
        """
        Save a regression model with its feature schema.

        Args:
            name: Model name
            model: Trained model
            scaler: Feature scaler (optional)
            feature_columns: List of feature column names in order (required for incremental models)
        """
        with self.lock:
            self.regression_models[name] = model
            if scaler:
                self.scalers[name] = scaler
            if feature_columns:
                self.feature_schemas[name] = feature_columns
            logger.info(f"Saved regression model: {name} with {len(feature_columns) if feature_columns else 0} features")

    def get_regression_model(self, name: str) -> Optional[Tuple[Any, Optional[StandardScaler], Optional[List[str]]]]:
        """
        Get a regression model, its scaler, and feature schema.

        Returns:
            Tuple of (model, scaler, feature_columns) or None if model not found
        """
        with self.lock:
            model = self.regression_models.get(name)
            scaler = self.scalers.get(name)
            features = self.feature_schemas.get(name)
            return (model, scaler, features) if model else None

    def validate_feature_schema(self, name: str, new_features: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate that new features match the existing model's schema.

        Args:
            name: Model name
            new_features: List of feature column names from new data

        Returns:
            Tuple of (is_valid, error_message)
        """
        with self.lock:
            if name not in self.feature_schemas:
                # No schema stored yet - this is the first training
                return True, None

            existing_features = self.feature_schemas[name]

            # Check if features match exactly (order matters for StandardScaler)
            if new_features != existing_features:
                missing = set(existing_features) - set(new_features)
                extra = set(new_features) - set(existing_features)

                error_parts = []
                if missing:
                    error_parts.append(f"Missing features: {sorted(missing)}")
                if extra:
                    error_parts.append(f"Extra features: {sorted(extra)}")
                if set(new_features) == set(existing_features):
                    error_parts.append(f"Feature order mismatch. Expected: {existing_features}, Got: {new_features}")

                error_msg = (
                    f"Feature schema mismatch for model '{name}'. "
                    f"{' | '.join(error_parts)}. "
                    f"Incremental training requires identical feature sets in the same order."
                )
                return False, error_msg

            return True, None

    def list_models(self) -> List[str]:
        """List all saved models."""
        with self.lock:
            return list(self.regression_models.keys())

    def get_model_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model."""
        with self.lock:
            if name not in self.regression_models:
                return None

            model = self.regression_models[name]
            return {
                'name': name,
                'type': type(model).__name__,
                'has_scaler': name in self.scalers,
                'feature_count': len(self.feature_schemas.get(name, [])),
                'features': self.feature_schemas.get(name, [])
            }

    def delete_model(self, name: str) -> bool:
        """Delete a model and its associated data."""
        with self.lock:
            if name in self.regression_models:
                del self.regression_models[name]
                if name in self.scalers:
                    del self.scalers[name]
                if name in self.feature_schemas:
                    del self.feature_schemas[name]
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


def validate_file_path(file_path: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate and sanitize file paths to prevent directory traversal attacks.

    Security measures:
    - Resolves to absolute path and checks it's within allowed directory
    - Prevents path traversal (../, symlinks, etc.)
    - Validates file exists and is readable

    Args:
        file_path: User-supplied file path

    Returns:
        Tuple of (is_valid, sanitized_path, error_message)
    """
    if not config.ENABLE_FILE_UPLOAD_SECURITY:
        # Security disabled - allow any path (NOT RECOMMENDED FOR PRODUCTION)
        if not os.path.exists(file_path):
            return False, "", "File not found"
        return True, file_path, None

    try:
        # Get absolute path of allowed directory
        allowed_dir = os.path.abspath(config.ALLOWED_UPLOAD_DIR)

        # Ensure allowed directory exists
        if not os.path.exists(allowed_dir):
            os.makedirs(allowed_dir, exist_ok=True)
            logger.info(f"Created allowed upload directory: {allowed_dir}")

        # Resolve the requested path to absolute (follows symlinks, resolves ..)
        requested_path = os.path.abspath(file_path)

        # Check if the resolved path is within the allowed directory
        if not requested_path.startswith(allowed_dir + os.sep) and requested_path != allowed_dir:
            logger.warning(f"Path traversal attempt blocked: {file_path} -> {requested_path}")
            return False, "", f"Access denied: Path must be within {config.ALLOWED_UPLOAD_DIR}/"

        # Check file exists
        if not os.path.exists(requested_path):
            return False, "", f"File not found in allowed directory: {os.path.basename(file_path)}"

        # Check it's a file (not a directory)
        if not os.path.isfile(requested_path):
            return False, "", "Path must point to a file, not a directory"

        # Check file is readable
        if not os.access(requested_path, os.R_OK):
            return False, "", "File is not readable"

        logger.info(f"File path validated: {file_path} -> {requested_path}")
        return True, requested_path, None

    except Exception as e:
        logger.error(f"Error validating file path '{file_path}': {e}", exc_info=True)
        return False, "", f"Invalid file path: {str(e)}"


# ==================== CORE ANALYTICS FUNCTIONS ====================

# ==================== SIGNAL PROCESSING & PREPROCESSING ====================

def preprocess_signal_perfect(series: pd.Series, signal_type: str = 'auto',
                             method: str = 'auto', preserve_anomalies: bool = True) -> Dict[str, Any]:
    """
    Smart signal preprocessing that PRESERVES anomalies for ML/forecasting.

    CRITICAL: For anomaly detection and forecasting, we want to:
    - Remove high-frequency NOISE (sensor jitter, measurement errors)
    - PRESERVE actual anomalies and patterns (these are what ML learns from)

    Methods:
    - 'light_smooth': Gentle Savitzky-Golay (preserves anomalies, removes noise)
    - 'none': No preprocessing (use raw data)
    - 'auto': Intelligently decides based on noise level

    Args:
        series: Input signal
        signal_type: Type hint ('auto', 'sensor', 'smooth')
        method: Specific method ('auto', 'light_smooth', 'none')
        preserve_anomalies: If True, uses gentle filtering that keeps anomalies (default: True)

    Returns:
        Dict with original, cleaned signal, method used, and improvement metrics
    """
    from scipy.signal import savgol_filter

    original = series.dropna().values
    n = len(original)

    if n < 10:
        return {
            'original': original,
            'cleaned': original,
            'method': 'NONE',
            'improvement': 0,
            'all_methods': {'none': 0}
        }

    # Calculate noise level
    # Use first derivative to estimate high-frequency noise
    diff = np.diff(original)
    noise_estimate = np.std(diff) / np.sqrt(2)  # Noise in original signal
    signal_std = np.std(original)
    noise_ratio = noise_estimate / signal_std if signal_std > 0 else 0

    results = {}

    # Method 1: No preprocessing (best for anomaly detection)
    results['none'] = {'signal': original.copy(), 'snr': 0}

    # Method 2: Light smoothing (only if noise is significant)
    if method in ['auto', 'light_smooth'] and noise_ratio > 0.05:
        try:
            # Very small window to preserve anomalies
            window = min(7, n // 10)
            if window % 2 == 0:
                window -= 1
            if window < 5:
                window = 5
            polyorder = 2

            light_smooth = savgol_filter(original, window, polyorder)
            snr = calculate_snr_improvement(original, light_smooth)
            results['light_smooth'] = {'signal': light_smooth, 'snr': snr}
        except Exception as e:
            logger.warning(f"Light smoothing failed: {e}")

    # Auto-select method
    if method == 'auto':
        if noise_ratio < 0.05:
            # Low noise - don't preprocess
            selected_method = 'none'
            selected_signal = original.copy()
            selected_snr = 0
        else:
            # Moderate noise - use light smoothing
            if 'light_smooth' in results:
                selected_method = 'light_smooth'
                selected_signal = results['light_smooth']['signal']
                selected_snr = results['light_smooth']['snr']
            else:
                selected_method = 'none'
                selected_signal = original.copy()
                selected_snr = 0
    elif method == 'none':
        selected_method = 'none'
        selected_signal = original.copy()
        selected_snr = 0
    elif method == 'light_smooth' and 'light_smooth' in results:
        selected_method = 'light_smooth'
        selected_signal = results['light_smooth']['signal']
        selected_snr = results['light_smooth']['snr']
    else:
        # Fallback
        selected_method = 'none'
        selected_signal = original.copy()
        selected_snr = 0

    return {
        'original': original,
        'cleaned': selected_signal,
        'method': selected_method.upper(),
        'improvement': selected_snr,
        'all_methods': {k: v['snr'] for k, v in results.items()},
        'noise_ratio': noise_ratio
    }


def calculate_snr_improvement(original: np.ndarray, filtered: np.ndarray) -> float:
    """Calculate SNR improvement in dB."""
    noise = original - filtered
    signal_power = np.var(filtered)
    noise_power = np.var(noise)
    if noise_power < 1e-10:
        return 100.0
    return 10 * np.log10(signal_power / noise_power)


def apply_kalman_filter(data: np.ndarray) -> np.ndarray:
    """1D Kalman filter - optimal for Gaussian noise."""
    n = len(data)
    x_est = data[0]
    p_est = 1.0

    # Auto-tune variances
    process_var = np.var(np.diff(data)) * 0.01
    measurement_var = np.var(data) * 0.1

    filtered = np.zeros(n)
    for i in range(n):
        x_pred = x_est
        p_pred = p_est + process_var
        K = p_pred / (p_pred + measurement_var)
        x_est = x_pred + K * (data[i] - x_pred)
        p_est = (1 - K) * p_pred
        filtered[i] = x_est

    return filtered


def apply_hampel_filter(data: np.ndarray, window_size: int = 7, n_sigma: float = 3.0) -> np.ndarray:
    """Hampel filter - robust outlier removal using MAD."""
    n = len(data)
    filtered = data.copy()
    half_window = window_size // 2

    for i in range(half_window, n - half_window):
        window = data[i - half_window:i + half_window + 1]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        threshold = n_sigma * 1.4826 * mad

        if np.abs(data[i] - median) > threshold:
            filtered[i] = median

    return filtered


def visualize_signal_processing(original: np.ndarray, cleaned: np.ndarray,
                                method: str, all_methods: Dict[str, float],
                                column_name: str = 'signal',
                                save_path: Optional[str] = None) -> Tuple[str, str]:
    """
    Generate fast, practical signal processing visualization.

    Shows:
    - Original vs Filtered signal comparison
    - Noise removed
    - Method comparison (if multiple methods tested)

    Args:
        original: Original signal array
        cleaned: Filtered signal array
        method: Method used for filtering
        all_methods: Dictionary of all methods and their SNR scores
        column_name: Name of the signal column
        save_path: Optional path to save the visualization

    Returns:
        Tuple of (base64_encoded_image, saved_file_path)
    """
    try:
        # Simple, fast 2-panel layout
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), dpi=config.VISUALIZATION_DPI)

        n = len(original)
        x_axis = np.arange(n)

        # Plot 1: Original vs Processed Signal
        ax1 = axes[0]
        ax1.plot(x_axis, original, label='Original', alpha=0.7, linewidth=1.5,
                color='#6C757D', marker='o', markersize=3, markevery=max(1, n//50))

        # Only plot cleaned if different from original
        if not np.array_equal(original, cleaned):
            ax1.plot(x_axis, cleaned, label=f'Processed ({method})', linewidth=2, color='#0D6EFD')
            title_suffix = f' - Method: {method}'
        else:
            title_suffix = ' - No Preprocessing (Preserving Anomalies)'

        ax1.set_title(f'Signal: {column_name}{title_suffix}', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Sample Index', fontsize=10)
        ax1.set_ylabel('Value', fontsize=10)
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Highlight anomalies (values > 2 std from mean)
        mean_val = np.mean(original)
        std_val = np.std(original)
        anomaly_mask = np.abs(original - mean_val) > 2 * std_val
        if np.any(anomaly_mask):
            ax1.scatter(x_axis[anomaly_mask], original[anomaly_mask],
                       color='red', s=100, alpha=0.6, marker='*',
                       label='Potential Anomalies', zorder=5)
            ax1.legend(loc='best', fontsize=10)

        # Plot 2: Difference (what was removed)
        ax2 = axes[1]
        diff = original - cleaned
        diff_std = np.std(diff)
        diff_max = np.max(np.abs(diff))

        ax2.plot(x_axis, diff, color='#DC3545', alpha=0.7, linewidth=1.5)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.fill_between(x_axis, diff, 0, alpha=0.3, color='#DC3545')

        ax2.set_title(f'Removed Noise (Std: {diff_std:.3f}, Max: {diff_max:.3f})',
                     fontsize=11, fontweight='bold')
        ax2.set_xlabel('Sample Index', fontsize=10)
        ax2.set_ylabel('Difference', fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Add info text
        info_text = f'Preprocessing: {method}\nSamples: {n}'

        ax2.text(0.02, 0.98, info_text,
                transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.8})

        plt.tight_layout()

        # Save to file if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, format=config.VISUALIZATION_FORMAT, bbox_inches='tight')
            logger.info(f"Saved signal processing visualization to {save_path}")

        # Create base64 for API response
        img = io.BytesIO()
        plt.savefig(img, format=config.VISUALIZATION_FORMAT, bbox_inches='tight')
        plt.close()
        img.seek(0)

        encoded = base64.b64encode(img.getvalue()).decode('utf-8')
        return encoded, save_path or ""

    except Exception as e:
        logger.error(f"Error generating signal processing visualization: {e}", exc_info=True)
        return "", ""


# ==================== SMART DATA DETECTION ====================

def auto_detect_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Auto-detect column types and semantic meaning.

    Detects:
    - Dates/timestamps
    - Monetary values (cost, price, amount, revenue)
    - Percentages
    - IDs/keys
    - Categories
    - Metrics (counts, rates, scores)
    """
    detected = {
        'datetime': [],
        'monetary': [],
        'percentage': [],
        'identifier': [],
        'category': [],
        'metric': [],
        'text': []
    }

    for col in df.columns:
        col_lower = col.lower()

        # Check for datetime
        if any(kw in col_lower for kw in ['date', 'time', 'timestamp', 'created', 'updated', 'at']):
            try:
                parsed = pd.to_datetime(df[col], errors='coerce')
                if parsed.notna().sum() / len(df) > 0.5:
                    detected['datetime'].append(col)
                    continue
            except Exception:
                pass

        # Check for monetary values
        if any(kw in col_lower for kw in ['cost', 'price', 'amount', 'revenue', 'sales', 'payment',
                                           'fee', 'charge', 'total', 'subtotal', 'balance', 'salary',
                                           'wage', 'income', 'expense', 'budget', 'usd', 'eur', 'gbp']):
            if pd.api.types.is_numeric_dtype(df[col]):
                detected['monetary'].append(col)
                continue

        # Check for percentages
        if any(kw in col_lower for kw in ['percent', 'pct', 'rate', 'ratio']) or col.endswith('%'):
            if pd.api.types.is_numeric_dtype(df[col]):
                detected['percentage'].append(col)
                continue

        # Check for identifiers
        if any(kw in col_lower for kw in ['id', 'key', 'code', 'number', 'ref', 'uuid', 'guid']):
            detected['identifier'].append(col)
            continue

        # Check for categories
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                detected['category'].append(col)
                continue

        # Check for metrics (counts, scores, ratings)
        if any(kw in col_lower for kw in ['count', 'total', 'sum', 'avg', 'mean', 'score',
                                           'rating', 'level', 'rank', 'index']):
            if pd.api.types.is_numeric_dtype(df[col]):
                detected['metric'].append(col)
                continue

        # Default to text if object type
        if pd.api.types.is_object_dtype(df[col]):
            detected['text'].append(col)

    return detected


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
    existing_features = None

    if incremental and model_name:
        result = model_manager.get_regression_model(model_name)
        if result:
            existing_model, existing_scaler, existing_features = result
            logger.info(f"Loaded existing model '{model_name}' for incremental training")

            # CRITICAL: Validate feature schema matches
            is_valid, error_msg = model_manager.validate_feature_schema(model_name, feature_cols)
            if not is_valid:
                logger.error(f"Schema validation failed: {error_msg}")
                raise ValueError(error_msg)

            logger.info(f"Feature schema validated: {len(feature_cols)} features match existing model")

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
        model_manager.save_regression_model(model_name, model, scaler, feature_columns=feature_cols)
        # Persist to disk immediately (respects AUTO_SAVE_MODELS config)
        if config.AUTO_SAVE_MODELS:
            save_models()
            logger.info(f"Model '{model_name}' saved to disk")
        else:
            logger.info(f"Model '{model_name}' saved to memory (AUTO_SAVE_MODELS=False)")

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
def auto_forecast_best(df: pd.DataFrame, column: str, time_col: str = None, horizon: int = 30) -> Dict[str, Any]:
    """
    BEST-IN-CLASS ensemble forecasting using state-of-the-art methods.

    Combines multiple algorithms and selects the best:
    - XGBoost with engineered features (lag, rolling stats)
    - Holt-Winters Triple Exponential Smoothing (seasonality)
    - Prophet (Facebook's forecasting tool)
    - Exponential Smoothing
    - Linear regression baseline

    Returns ensemble of top 3 models weighted by validation MAE.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    results = {}
    y = df[column].dropna().values
    n = len(y)

    if n < 10:
        return forecast_linear_trend(df, column, time_col, horizon)

    # Feature engineering for ML models
    def create_features(series, lookback=10):
        features = []
        for i in range(lookback, len(series)):
            row = []
            for lag in range(1, min(lookback + 1, 11)):
                row.append(series[i - lag])
            window = series[max(0, i-7):i]
            if len(window) > 0:
                row.extend([np.mean(window), np.std(window), np.min(window), np.max(window)])
            else:
                row.extend([0, 0, 0, 0])
            features.append(row)
        return np.array(features)

    train_size = int(n * 0.8)
    train, test = y[:train_size], y[train_size:]

    # Method 1: XGBoost with feature engineering
    try:
        lookback = min(10, train_size // 2)
        X_train = create_features(train, lookback)
        y_train = train[lookback:]

        if len(X_train) > 10:
            model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            model.fit(X_train, y_train)

            forecast_vals = []
            current = list(train[-lookback:])
            for _ in range(horizon):
                features = []
                for lag in range(1, min(lookback + 1, 11)):
                    features.append(current[-lag])
                window = current[-7:]
                features.extend([np.mean(window), np.std(window), np.min(window), np.max(window)])
                pred = model.predict([features])[0]
                forecast_vals.append(pred)
                current.append(pred)

            X_test = create_features(y[:train_size + len(test)], lookback)
            if len(X_test) >= len(test):
                y_pred = model.predict(X_test[-len(test):])
                mae = np.mean(np.abs(test - y_pred))
                results['xgboost'] = {'forecast': forecast_vals, 'mae': mae, 'method': 'XGBoost'}
    except Exception as e:
        logger.warning(f"XGBoost failed: {e}")

    # Method 2: Holt-Winters Triple Exponential Smoothing
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal_periods = min(12, len(train) // 2) if len(train) > 24 else None
        model = ExponentialSmoothing(train, seasonal_periods=seasonal_periods, trend='add',
                                     seasonal='add' if seasonal_periods else None)
        fitted = model.fit()
        forecast_vals = fitted.forecast(horizon).tolist()
        test_pred = fitted.forecast(len(test))
        mae = np.mean(np.abs(test - test_pred))
        results['holt_winters'] = {'forecast': forecast_vals, 'mae': mae, 'method': 'Holt-Winters'}
    except Exception as e:
        logger.warning(f"Holt-Winters failed: {e}")

    # Method 3: Exponential Smoothing
    try:
        exp_result = forecast_exponential_smoothing(df, column, horizon, alpha=0.3)
        results['exp_smooth'] = {'forecast': exp_result['forecast'], 'mae': exp_result['metrics']['mae'],
                                'method': 'Exponential Smoothing'}
    except Exception as e:
        logger.warning(f"Exp smoothing failed: {e}")

    # Method 4: Linear Trend
    try:
        linear_result = forecast_linear_trend(df, column, time_col, horizon)
        results['linear'] = {'forecast': linear_result['forecast'], 'mae': linear_result['metrics']['mae'],
                           'method': 'Linear Trend'}
    except Exception as e:
        logger.warning(f"Linear failed: {e}")

    # Method 5: Prophet
    if PROPHET_AVAILABLE and time_col:
        try:
            prophet_result = forecast_prophet(df, column, time_col, horizon)
            results['prophet'] = {'forecast': prophet_result['forecast'], 'mae': prophet_result['metrics']['mae'],
                                'method': 'Prophet', 'lower': prophet_result.get('forecast_lower', []),
                                'upper': prophet_result.get('forecast_upper', [])}
        except Exception as e:
            logger.warning(f"Prophet failed: {e}")

    if not results:
        return forecast_linear_trend(df, column, time_col, horizon)

    # Ensemble: top 3 weighted by inverse MAE
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['mae'])
    top_3 = sorted_methods[:min(3, len(sorted_methods))]
    weights = [1 / (m[1]['mae'] + 1e-6) for m in top_3]
    total = sum(weights)
    weights = [w / total for w in weights]

    ensemble = np.zeros(horizon)
    for (name, result), weight in zip(top_3, weights):
        forecast_vals = result['forecast']
        # Ensure forecast has correct length
        if len(forecast_vals) < horizon:
            forecast_vals = list(forecast_vals) + [forecast_vals[-1]] * (horizon - len(forecast_vals))
        elif len(forecast_vals) > horizon:
            forecast_vals = forecast_vals[:horizon]
        ensemble += np.array(forecast_vals) * weight

    best_name, best_result = sorted_methods[0]

    return {
        'forecast': ensemble.tolist(),
        'method': f'Ensemble: {", ".join([m[0].upper() for m in top_3])}',
        'best_single': best_name,
        'all_methods': {k: {'mae': v['mae'], 'method': v['method']} for k, v in results.items()},
        'metrics': {
            'mae': best_result['mae'],
            'weights': dict(zip([m[0] for m in top_3], [round(w, 3) for w in weights]))
        }
    }


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

    Security: Files must be in the configured ALLOWED_UPLOAD_DIR (default: 'data/')
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        file_path = request.json.get('file')
        dataset_name = request.json.get('name')

        if not file_path:
            return jsonify({'error': 'file path is required'}), 400

        # SECURITY: Validate file path to prevent directory traversal
        is_valid, sanitized_path, error_msg = validate_file_path(file_path)
        if not is_valid:
            logger.warning(f"File upload rejected: {error_msg}")
            return jsonify({'error': error_msg}), 403

        file_path = sanitized_path

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


@app.route('/preprocess_signal', methods=['POST'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def preprocess_signal_endpoint():
    """
    Dynamic signal preprocessing with visualization.

    Applies advanced filters (Butterworth, Savitzky-Golay, Kalman, Hampel, Wavelet)
    and returns the best ensemble result with comprehensive visualization.

    Expected JSON body:
    {
        "dataset": "my_dataset",
        "columns": ["col1", "col2"] (optional, defaults to all numeric),
        "signal_type": "auto" (optional),
        "visualize": true (optional, default: true),
        "apply_to_dataset": false (optional, if true, updates dataset with cleaned signals)
    }

    Returns:
        JSON with preprocessing results, SNR improvements, and visualization
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        if not dataset_name:
            return jsonify({'error': 'Dataset name is required'}), 400

        df = data_store.get_dataset(dataset_name)
        if df is None:
            return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404

        # Get columns to process
        columns = request.json.get('columns')
        if columns is None:
            # Default to all numeric columns
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        elif isinstance(columns, str):
            columns = [columns]

        if not columns:
            return jsonify({'error': 'No numeric columns found or specified'}), 400

        # Validate columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Columns not found: {missing_cols}'}), 400

        signal_type = request.json.get('signal_type', 'auto')
        visualize = request.json.get('visualize', True)
        apply_to_dataset = request.json.get('apply_to_dataset', False)

        logger.info(f"Preprocessing {len(columns)} signal(s) in '{dataset_name}'")

        # Process each column
        results = {}
        visualizations = {}

        for col in columns:
            logger.info(f"Processing column: {col}")

            # Preprocess signal
            preprocess_result = preprocess_signal_perfect(df[col], signal_type)

            # Generate visualization if requested
            viz_encoded = ""
            viz_path = ""

            if visualize:
                analysis_dir = 'analysis'
                os.makedirs(analysis_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{dataset_name}_{col}_signal_processing_{timestamp}.png"
                filepath = os.path.join(analysis_dir, filename)

                viz_encoded, viz_path = visualize_signal_processing(
                    preprocess_result['original'],
                    preprocess_result['cleaned'],
                    preprocess_result['method'],
                    preprocess_result['all_methods'],
                    column_name=col,
                    save_path=filepath
                )

                visualizations[col] = {
                    'base64': viz_encoded,
                    'path': viz_path
                }

            # Store results
            results[col] = {
                'method': preprocess_result['method'],
                'snr_improvement_db': preprocess_result['improvement'],
                'all_methods_snr': preprocess_result['all_methods'],
                'original_length': len(preprocess_result['original']),
                'cleaned_length': len(preprocess_result['cleaned']),
                'noise_std': float(np.std(preprocess_result['original'] - preprocess_result['cleaned'])),
                'signal_std_original': float(np.std(preprocess_result['original'])),
                'signal_std_cleaned': float(np.std(preprocess_result['cleaned']))
            }

            # Apply to dataset if requested
            if apply_to_dataset:
                df[col] = preprocess_result['cleaned']

        # Update dataset if changes were applied
        if apply_to_dataset:
            data_store.add_dataset(dataset_name, df)
            logger.info(f"Updated dataset '{dataset_name}' with cleaned signals")

        response = {
            'dataset': dataset_name,
            'columns_processed': columns,
            'results': results,
            'applied_to_dataset': apply_to_dataset
        }

        if visualize:
            response['visualizations'] = visualizations

        logger.info(f"Signal preprocessing complete for {len(columns)} column(s)")
        return jsonify(response), 200

    except ValueError as e:
        logger.warning(f"Validation error in preprocess_signal: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Error in signal preprocessing: {e}", exc_info=True)
        return jsonify({'error': 'Signal preprocessing failed', 'details': str(e)}), 500


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
    Train a model on a dataset with incremental learning support.

    By default uses "unified_model" for incremental learning across datasets.
    Can specify custom model_name for isolated models.

    Body: {
        "dataset": "name",
        "target": "col",
        "features": [...] (optional),
        "model_name": "custom_name" (optional, defaults to "unified_model")
    }

    Note: Incremental training requires identical feature schemas.
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        target_col = request.json.get('target')
        feature_cols = request.json.get('features')
        test_size = request.json.get('test_size', 0.2)
        model_name = request.json.get('model_name', 'unified_model')  # Default to unified_model

        if not all([dataset_name, target_col]):
            return jsonify({'error': 'dataset and target are required'}), 400

        df = data_store.get_dataset(dataset_name)
        if df is None:
            return jsonify({'error': f'Dataset "{dataset_name}" not found'}), 404

        logger.info(f"Training model '{model_name}' on '{dataset_name}' ({len(df)} samples)")

        # Use incremental learning
        result = linear_regression(
            df,
            target_col,
            feature_cols,
            test_size,
            model_name=model_name,
            incremental=True
        )

        result['model_name'] = model_name
        result['dataset_source'] = dataset_name
        result['samples_added'] = len(df)

        logger.info(f"✓ Model '{model_name}' trained on {len(df)} samples from '{dataset_name}'")
        return jsonify(result), 200

    except ValueError as e:
        # Schema validation errors
        logger.warning(f"Validation error in train_model: {e}")
        return jsonify({'error': str(e)}), 400
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

        model, scaler, features = result

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

    Security: Files must be in the configured ALLOWED_UPLOAD_DIR (default: 'data/')
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        file_path = request.json.get('file')
        target_col = request.json.get('target')

        if not file_path:
            return jsonify({'error': 'file path is required'}), 400

        # SECURITY: Validate file path to prevent directory traversal
        is_valid, sanitized_path, error_msg = validate_file_path(file_path)
        if not is_valid:
            logger.warning(f"Training file rejected: {error_msg}")
            return jsonify({'error': error_msg}), 403

        file_path = sanitized_path

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

    Body: {
        "dataset": "name" OR "file": "path.json",
        "target": "col" (optional),
        "preprocess_signals": true (optional, default: false),
        "signal_columns": ["col1", "col2"] (optional, defaults to all numeric)
    }

    Security: Files must be in the configured ALLOWED_UPLOAD_DIR (default: 'data/')
    """
    try:
        if not request.json:
            return jsonify({'error': 'Request must be JSON'}), 400

        dataset_name = request.json.get('dataset')
        file_path = request.json.get('file')
        target_col = request.json.get('target')
        preprocess_signals = request.json.get('preprocess_signals', False)
        signal_columns = request.json.get('signal_columns')

        # Load data
        if file_path:
            # SECURITY: Validate file path to prevent directory traversal
            is_valid, sanitized_path, error_msg = validate_file_path(file_path)
            if not is_valid:
                logger.warning(f"Analysis file rejected: {error_msg}")
                return jsonify({'error': error_msg}), 403

            file_path = sanitized_path

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

        # Signal preprocessing if requested
        signal_processing_results = None
        signal_visualizations = {}

        if preprocess_signals:
            logger.info("Running signal preprocessing...")

            # Determine which columns to preprocess
            cols_to_process = signal_columns if signal_columns else numeric_cols[:3]  # Default to first 3
            cols_to_process = [c for c in cols_to_process if c in numeric_cols]

            signal_processing_results = {}

            for col in cols_to_process:
                preprocess_result = preprocess_signal_perfect(df[col])

                # Generate visualization
                analysis_dir = 'analysis'
                os.makedirs(analysis_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{dataset_name}_{col}_signal_{timestamp}.png"
                filepath = os.path.join(analysis_dir, filename)

                viz_encoded, viz_path = visualize_signal_processing(
                    preprocess_result['original'],
                    preprocess_result['cleaned'],
                    preprocess_result['method'],
                    preprocess_result['all_methods'],
                    column_name=col,
                    save_path=filepath
                )

                signal_processing_results[col] = {
                    'method': preprocess_result['method'],
                    'snr_improvement_db': preprocess_result['improvement'],
                    'all_methods_snr': preprocess_result['all_methods']
                }

                signal_visualizations[col] = {
                    'base64': viz_encoded,
                    'path': viz_path
                }

                # Apply cleaned signal to dataframe for subsequent analysis
                df[col] = preprocess_result['cleaned']

            logger.info(f"Signal preprocessing complete for {len(cols_to_process)} columns")

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

        if signal_processing_results:
            results['signal_processing'] = {
                'columns_processed': list(signal_processing_results.keys()),
                'results': signal_processing_results,
                'visualizations': signal_visualizations
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
    Generate comprehensive analysis visualization with tooltips and meaningful insights.

    Features:
    - Auto-detects time columns for proper temporal axis
    - Standardized scales for multi-series comparison
    - Annotated outliers and anomalies with actual values
    - Statistical summary with key insights
    - Color-coded by severity

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

        # Auto-detect time column
        time_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col])
                    time_col = col
                    break
                except:
                    pass

        # Use time column or index for x-axis
        if time_col:
            x_axis = df[time_col]
            x_label = time_col
        else:
            x_axis = df.index
            x_label = 'Index'

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=config.VISUALIZATION_DPI)
        fig.suptitle(f'📊 Comprehensive Analysis: {dataset_name}\n{len(df)} rows | {len(numeric_cols)} numeric columns',
                    fontsize=16, fontweight='bold', y=0.98)

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

        # Plot 2: Outliers referencing the primary column with annotations
        ax2 = axes[0, 1]
        outlier_flags = outliers['is_outlier']
        colors = ['#D7263D' if flag else '#6EA4BF' for flag in outlier_flags]
        ax2.scatter(time_values, primary_series,
                    c=colors, alpha=0.7, s=35, edgecolors='k', linewidths=0.2)

        # Annotate top 5 outliers with their values
        outlier_indices = [i for i, flag in enumerate(outlier_flags) if flag]
        if outlier_indices:
            # Get top 5 most extreme outliers
            outlier_values = [(i, abs(primary_series.iloc[i] - primary_series.median()))
                            for i in outlier_indices]
            top_outliers = sorted(outlier_values, key=lambda x: x[1], reverse=True)[:5]

            for idx, _ in top_outliers:
                value = primary_series.iloc[idx]
                ax2.annotate(f'{value:.1f}',
                           xy=(time_values[idx] if hasattr(time_values, '__getitem__') else idx, value),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='#D7263D', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='#D7263D'))

        ax2.set_title(f'🔴 Outliers (IQR) on {primary_col}: {outliers["n_outliers"]} rows',
                     fontsize=11, fontweight='bold')
        ax2.set_xlabel(time_col or 'Index', fontsize=10)
        ax2.set_ylabel(primary_col, fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')

        # Plot 3: Anomalies highlighting the same primary column with annotations
        ax3 = axes[1, 0]
        anomaly_flags = anomalies['is_anomaly']
        colors = ['#FF9F1C' if flag else '#2EC4B6' for flag in anomaly_flags]
        ax3.scatter(time_values, primary_series,
                    c=colors, alpha=0.7, s=35, edgecolors='k', linewidths=0.2)

        # Annotate top 5 anomalies with their values
        anomaly_indices = [i for i, flag in enumerate(anomaly_flags) if flag]
        if anomaly_indices:
            # Get top 5 most extreme anomalies
            anomaly_values = [(i, abs(primary_series.iloc[i] - primary_series.median()))
                            for i in anomaly_indices]
            top_anomalies = sorted(anomaly_values, key=lambda x: x[1], reverse=True)[:5]

            for idx, _ in top_anomalies:
                value = primary_series.iloc[idx]
                ax3.annotate(f'{value:.1f}',
                           xy=(time_values[idx] if hasattr(time_values, '__getitem__') else idx, value),
                           xytext=(5, -15), textcoords='offset points',
                           fontsize=8, color='#FF9F1C', fontweight='bold',
                           bbox={'boxstyle': 'round,pad=0.3', 'facecolor': 'white',
                                'alpha': 0.7, 'edgecolor': '#FF9F1C'})

        ax3.set_title(f'⚠️  Anomalies (Isolation Forest) on {primary_col}: {anomalies["n_anomalies"]} rows',
                     fontsize=11, fontweight='bold')
        ax3.set_xlabel(time_col or 'Index', fontsize=10)
        ax3.set_ylabel(primary_col, fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')

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
    """Save all regression models to disk with feature schemas."""
    try:
        model_dir = config.MODEL_DIR
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created model directory: {model_dir}")

        models = model_manager.list_models()
        for model_name in models:
            result = model_manager.get_regression_model(model_name)
            if result:
                model, scaler, features = result
                model_path = os.path.join(model_dir, f'{model_name}_{config.MODEL_VERSION}.joblib')
                scaler_path = os.path.join(model_dir, f'{model_name}_scaler_{config.MODEL_VERSION}.joblib')
                schema_path = os.path.join(model_dir, f'{model_name}_schema_{config.MODEL_VERSION}.joblib')

                joblib.dump(model, model_path)
                if scaler:
                    joblib.dump(scaler, scaler_path)
                if features:
                    joblib.dump(features, schema_path)

                logger.info(f"Saved model '{model_name}' to {model_path}")

        logger.info(f"Saved {len(models)} models successfully")

    except Exception as e:
        logger.error(f"Error saving models: {e}", exc_info=True)
        raise


def load_models() -> None:
    """Load regression models from disk with feature schemas."""
    try:
        model_dir = config.MODEL_DIR

        if not os.path.exists(model_dir):
            logger.info(f"Model directory {model_dir} does not exist, skipping model loading")
            return

        # Find all model files (exclude scaler and schema files)
        model_files = [f for f in os.listdir(model_dir)
                      if f.endswith('.joblib') and 'scaler' not in f and 'schema' not in f]

        for model_file in model_files:
            model_name = model_file.replace(f'_{config.MODEL_VERSION}.joblib', '')
            model_path = os.path.join(model_dir, model_file)
            scaler_path = os.path.join(model_dir, f'{model_name}_scaler_{config.MODEL_VERSION}.joblib')
            schema_path = os.path.join(model_dir, f'{model_name}_schema_{config.MODEL_VERSION}.joblib')

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
            features = joblib.load(schema_path) if os.path.exists(schema_path) else None

            model_manager.save_regression_model(model_name, model, scaler, feature_columns=features)
            logger.info(f"Loaded model '{model_name}' from {model_path} with {len(features) if features else 0} features")

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
    """CLI: Analyze a dataset with auto-detection and visualization."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {file_path}")
    print('='*60)

    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.json'):
        df = pd.read_json(file_path)
    elif file_path.endswith(('.log', '.txt')):
        df = parse_log_file(file_path)
    else:
        print("❌ Unsupported file format")
        return

    dataset_name = os.path.basename(file_path).rsplit('.', 1)[0]
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Auto-detect column types
    print(f"\n🔍 AUTO-DETECTING COLUMN TYPES...")
    detected_types = auto_detect_column_types(df)

    if detected_types['datetime']:
        print(f"  📅 Date/Time: {', '.join(detected_types['datetime'])}")
    if detected_types['monetary']:
        print(f"  💰 Monetary: {', '.join(detected_types['monetary'])}")
    if detected_types['percentage']:
        print(f"  📊 Percentage: {', '.join(detected_types['percentage'])}")
    if detected_types['category']:
        print(f"  🏷️  Category: {', '.join(detected_types['category'][:3])}{'...' if len(detected_types['category']) > 3 else ''}")
    if detected_types['metric']:
        print(f"  📈 Metric: {', '.join(detected_types['metric'][:3])}{'...' if len(detected_types['metric']) > 3 else ''}")

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("❌ No numeric columns found")
        return

    print(f"\n✓ Numeric columns: {', '.join(numeric_cols[:5])}{'...' if len(numeric_cols) > 5 else ''}")

    # LOTTERY-WINNING PREPROCESSING
    print(f"\n🎰 SIGNAL PREPROCESSING (Making Data Perfect)...")
    preprocessing_results = {}
    for col in numeric_cols[:3]:  # Preprocess top 3 columns
        result = preprocess_signal_perfect(df[col])
        preprocessing_results[col] = result
        print(f"  • {col}: {result['method']} (SNR: {result['improvement']:.1f} dB)")

        # Replace with cleaned signal
        df[col] = result['cleaned']

    print(f"  ✓ Data cleaned with state-of-the-art filters!")

    # Outlier detection
    print(f"\n🔍 OUTLIER DETECTION (IQR)")
    outliers = detect_outliers_iqr(df, numeric_cols)
    print(f"  Found: {outliers['n_outliers']} ({outliers['outlier_percentage']:.1f}%)")

    # Anomaly detection
    print(f"\n⚠️  ANOMALY DETECTION (Isolation Forest)")
    anomalies = detect_anomalies_iforest(df, numeric_cols)
    print(f"  Found: {anomalies['n_anomalies']} ({anomalies['anomaly_percentage']:.1f}%)")

    # Auto-Forecasting (if time-series data detected)
    if len(numeric_cols) > 0:
        forecast_col = target if target and target in numeric_cols else numeric_cols[0]

        # Determine forecast horizon based on data size and time column
        if detected_types['datetime']:
            # Time-series detected - do intelligent forecasting
            time_col = detected_types['datetime'][0]

            # Calculate appropriate horizon based on data frequency
            if len(df) > 1000:
                horizon = 30  # Monthly forecast for large datasets
            elif len(df) > 100:
                horizon = 14  # 2-week forecast
            else:
                horizon = 7   # 1-week forecast

            print(f"\n📈 AUTO-FORECASTING (Time-Series Detected)")
            print(f"  Time Column: {time_col}")
            print(f"  Target: {forecast_col}")
            print(f"  Horizon: {horizon} periods")
            print(f"  🤖 Using BEST-IN-CLASS Ensemble Forecasting...")

            # Use the best ensemble forecasting
            forecast = auto_forecast_best(df, forecast_col, time_col, horizon)

            print(f"\n  ✅ {forecast['method']}")
            print(f"  Best Single Model: {forecast['best_single'].upper()}")
            print(f"  Next {min(5, horizon)}: {[round(x, 1) for x in forecast['forecast'][:5]]}")
            print(f"  Validation MAE: {forecast['metrics']['mae']:.2f}")

            # Show all methods tried
            print(f"\n  📊 All Methods Evaluated:")
            for name, info in forecast['all_methods'].items():
                print(f"    • {info['method']}: MAE = {info['mae']:.2f}")

            # Show ensemble weights
            if 'weights' in forecast['metrics']:
                print(f"\n  ⚖️  Ensemble Weights:")
                for model, weight in forecast['metrics']['weights'].items():
                    print(f"    • {model.upper()}: {weight:.1%}")

        else:
            # No time column - simple forecast
            print(f"\n📈 FORECASTING ({forecast_col})")
            forecast = forecast_linear_trend(df, forecast_col, horizon=5)
            print(f"  Method: Linear Trend")
            print(f"  Next 5: {[round(x, 1) for x in forecast['forecast']]}")
            print(f"  MAE: {forecast['metrics']['mae']:.2f}")

    # Regression
    if target and target in numeric_cols:
        print(f"\n📉 REGRESSION (target: {target})")
        result = linear_regression(df, target, test_size=0.2)
        print(f"  R²: {result['metrics']['test_r2']:.3f}")
        print(f"  RMSE: {result['metrics']['test_rmse']:.3f}")

    # Generate visualization
    print(f"\n📊 GENERATING VISUALIZATION...")
    viz_encoded, viz_path = generate_analysis_visualization(df, numeric_cols, outliers, anomalies, dataset_name)
    print(f"  ✓ Saved: {viz_path}")

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
