"""
Production-ready Log Analysis API with ML-based anomaly detection and pattern recognition.
"""
from flask import Flask, request, jsonify, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_cors import CORS
from prometheus_flask_exporter import PrometheusMetrics
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from pyod.models.iforest import IForest
from collections import Counter, deque
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
import nltk
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
from typing import List, Dict, Optional, Any, Tuple
from threading import Lock
from functools import wraps
from dotenv import load_dotenv

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

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

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
class LogDataStore:
    """Thread-safe log data storage with memory management."""

    def __init__(self, max_size: int, retention_hours: int):
        self.max_size = max_size
        self.retention_hours = retention_hours
        self.logs = deque(maxlen=max_size)
        self.lock = Lock()
        logger.info(f"Initialized LogDataStore with max_size={max_size}, retention_hours={retention_hours}")

    def add_logs(self, logs: List[Dict[str, Any]]) -> int:
        """Add logs to the store with thread safety."""
        with self.lock:
            initial_size = len(self.logs)
            self.logs.extend(logs)
            added = len(self.logs) - initial_size
            logger.debug(f"Added {added} logs to store. Total: {len(self.logs)}")
            return added

    def get_logs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get logs from the store."""
        with self.lock:
            if limit:
                return list(self.logs)[-limit:]
            return list(self.logs)

    def cleanup_old_logs(self) -> int:
        """Remove logs older than retention period."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
            initial_size = len(self.logs)

            # Filter out old logs
            self.logs = deque(
                (log for log in self.logs if log.get('timestamp', datetime.now()) > cutoff_time),
                maxlen=self.max_size
            )

            removed = initial_size - len(self.logs)
            if removed > 0:
                logger.info(f"Cleaned up {removed} old logs")
            return removed

    def clear(self) -> None:
        """Clear all logs."""
        with self.lock:
            self.logs.clear()
            logger.info("Cleared all logs from store")

    def size(self) -> int:
        """Get current size of log store."""
        with self.lock:
            return len(self.logs)

# Global data store and models
log_store = LogDataStore(config.MAX_LOGS_IN_MEMORY, config.LOG_RETENTION_HOURS)
model_lock = Lock()

# Model storage
class ModelManager:
    """Thread-safe model management."""

    def __init__(self):
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.anomaly_model: Optional[IForest] = None
        self.pattern_model: Optional[DBSCAN] = None
        self.word2vec_model: Optional[Word2Vec] = None
        self.lock = Lock()
        logger.info("Initialized ModelManager")


    def get_vectorizer(self) -> Optional[TfidfVectorizer]:
        with self.lock:
            return self.vectorizer

    def set_vectorizer(self, vectorizer: TfidfVectorizer) -> None:
        with self.lock:
            self.vectorizer = vectorizer

    def get_anomaly_model(self) -> Optional[IForest]:
        with self.lock:
            return self.anomaly_model

    def set_anomaly_model(self, model: IForest) -> None:
        with self.lock:
            self.anomaly_model = model

    def get_pattern_model(self) -> Optional[DBSCAN]:
        with self.lock:
            return self.pattern_model

    def set_pattern_model(self, model: DBSCAN) -> None:
        with self.lock:
            self.pattern_model = model

    def get_word2vec_model(self) -> Optional[Word2Vec]:
        with self.lock:
            return self.word2vec_model

    def set_word2vec_model(self, model: Word2Vec) -> None:
        with self.lock:
            self.word2vec_model = model

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


# Input validation functions
def validate_log_format(log: str) -> bool:
    """Validate log entry format."""
    if not isinstance(log, str):
        return False

    log = log.strip()
    if not log or len(log) < 10:
        return False

    parts = log.split(' ', 2)
    return len(parts) >= 3


def sanitize_log(log: str) -> str:
    """Sanitize log entry to prevent injection attacks."""
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in log if ord(char) >= 32 or char in '\t\n\r')
    return sanitized.strip()


def preprocess_log(log: str) -> Optional[Dict[str, Any]]:
    """
    Preprocess a log entry into structured format.

    Args:
        log: Raw log string in format "timestamp level message"

    Returns:
        Dictionary with timestamp, level, and message, or None if invalid
    """
    try:
        if not validate_log_format(log):
            logger.debug(f"Invalid log format: {log[:50]}...")
            return None

        log = sanitize_log(log)
        parts = log.split(' ', 2)

        if len(parts) < 3:
            return None

        timestamp_str, level, message = parts

        # Parse timestamp
        try:
            timestamp = pd.to_datetime(timestamp_str)
        except Exception as e:
            logger.debug(f"Failed to parse timestamp '{timestamp_str}': {e}")
            # Use current time as fallback
            timestamp = pd.Timestamp.now()

        # Validate log level
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'WARN', 'FATAL'}
        if level.upper() not in valid_levels:
            logger.debug(f"Unknown log level: {level}")

        return {
            'timestamp': timestamp,
            'level': level.upper(),
            'message': message,
            'raw': log
        }

    except Exception as e:
        logger.error(f"Error preprocessing log: {e}", exc_info=True)
        return None


@track_time
def extract_features(messages: List[str]) -> np.ndarray:
    """
    Extract TF-IDF features from log messages.

    Args:
        messages: List of log messages

    Returns:
        Feature matrix (dense numpy array)
    """
    try:
        vectorizer = model_manager.get_vectorizer()

        if vectorizer is None:
            logger.info(f"Creating new TfidfVectorizer with max_features={config.TFIDF_MAX_FEATURES}")
            vectorizer = TfidfVectorizer(
                max_features=config.TFIDF_MAX_FEATURES,
                strip_accents='unicode',
                lowercase=True,
                stop_words='english'
            )
            features = vectorizer.fit_transform(messages)
            model_manager.set_vectorizer(vectorizer)
        else:
            features = vectorizer.transform(messages)

        # Convert to dense array
        if scipy.sparse.issparse(features):
            features = features.toarray()

        logger.debug(f"Extracted features shape: {features.shape}")
        return features

    except Exception as e:
        logger.error(f"Error extracting features: {e}", exc_info=True)
        raise


@track_time
def detect_anomalies(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect anomalies using Isolation Forest.

    Args:
        features: Feature matrix

    Returns:
        Tuple of (predictions, anomaly_scores)
    """
    try:
        anomaly_model = model_manager.get_anomaly_model()

        if anomaly_model is None:
            logger.info(f"Creating new IForest model with contamination={config.ANOMALY_CONTAMINATION}")
            anomaly_model = IForest(
                contamination=config.ANOMALY_CONTAMINATION,
                random_state=config.IFOREST_RANDOM_STATE,
                n_estimators=100
            )

        # Convert sparse matrix to dense array if necessary
        if scipy.sparse.issparse(features):
            features = features.toarray()

        anomaly_model.fit(features)
        model_manager.set_anomaly_model(anomaly_model)

        predictions = anomaly_model.predict(features)
        scores = anomaly_model.decision_function(features)

        logger.debug(f"Detected {np.sum(predictions == 1)} anomalies out of {len(predictions)} logs")
        return predictions, scores

    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}", exc_info=True)
        raise


@track_time
def find_patterns(features: np.ndarray) -> np.ndarray:
    """
    Find patterns using DBSCAN clustering.

    Args:
        features: Feature matrix

    Returns:
        Cluster labels
    """
    try:
        pattern_model = model_manager.get_pattern_model()

        if pattern_model is None:
            logger.info(f"Creating new DBSCAN model with eps={config.DBSCAN_EPS}, min_samples={config.DBSCAN_MIN_SAMPLES}")
            pattern_model = DBSCAN(
                eps=config.DBSCAN_EPS,
                min_samples=config.DBSCAN_MIN_SAMPLES,
                metric='cosine'
            )

        # Convert sparse matrix to dense array if necessary
        if scipy.sparse.issparse(features):
            features = features.toarray()

        labels = pattern_model.fit_predict(features)
        model_manager.set_pattern_model(pattern_model)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        logger.debug(f"Found {n_clusters} clusters and {n_noise} noise points")

        return labels

    except Exception as e:
        logger.error(f"Error finding patterns: {e}", exc_info=True)
        raise


@track_time
def nlp_analysis(messages: List[str]) -> List[np.ndarray]:
    """
    Perform NLP analysis using Word2Vec.

    Args:
        messages: List of log messages

    Returns:
        List of message vectors
    """
    try:
        word2vec_model = model_manager.get_word2vec_model()

        if word2vec_model is None or len(messages) > 1000:
            logger.info("Training new Word2Vec model")
            # Preprocess messages
            preprocessed_messages = []
            for msg in messages:
                # Extract just the message part (skip timestamp and level)
                parts = msg.split(' ', 2)
                if len(parts) >= 3:
                    preprocessed_messages.append(simple_preprocess(parts[2]))
                else:
                    preprocessed_messages.append(simple_preprocess(msg))

            word2vec_model = Word2Vec(
                sentences=preprocessed_messages,
                vector_size=config.WORD2VEC_VECTOR_SIZE,
                window=config.WORD2VEC_WINDOW,
                min_count=config.WORD2VEC_MIN_COUNT,
                workers=4,
                epochs=10
            )
            model_manager.set_word2vec_model(word2vec_model)

        def get_message_vector(msg: str) -> np.ndarray:
            """Get vector representation of a message."""
            parts = msg.split(' ', 2)
            text = parts[2] if len(parts) >= 3 else msg
            words = simple_preprocess(text)

            word_vectors = [
                word2vec_model.wv[word] for word in words
                if word in word2vec_model.wv
            ]

            if word_vectors:
                return np.mean(word_vectors, axis=0)
            else:
                return np.zeros(word2vec_model.vector_size)

        return [get_message_vector(msg) for msg in messages]

    except Exception as e:
        logger.error(f"Error in NLP analysis: {e}", exc_info=True)
        # Return zero vectors as fallback
        return [np.zeros(config.WORD2VEC_VECTOR_SIZE) for _ in messages]


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
        log_count = log_store.size()

        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': config.MODEL_VERSION,
            'log_count': log_count,
            'models_loaded': {
                'vectorizer': model_manager.get_vectorizer() is not None,
                'anomaly_model': model_manager.get_anomaly_model() is not None,
                'pattern_model': model_manager.get_pattern_model() is not None,
                'word2vec_model': model_manager.get_word2vec_model() is not None
            }
        }

        return jsonify(health_status), 200

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500


@app.route('/status', methods=['GET'])
def status():
    """Get detailed system status."""
    try:
        log_count = log_store.size()

        status_info = {
            'timestamp': datetime.now().isoformat(),
            'version': config.MODEL_VERSION,
            'configuration': {
                'max_logs_in_memory': config.MAX_LOGS_IN_MEMORY,
                'max_batch_size': config.MAX_BATCH_SIZE,
                'log_retention_hours': config.LOG_RETENTION_HOURS,
                'rate_limit_enabled': config.RATE_LIMIT_ENABLED
            },
            'statistics': {
                'total_logs': log_count,
                'memory_usage_percent': (log_count / config.MAX_LOGS_IN_MEMORY) * 100
            },
            'models': {
                'vectorizer_loaded': model_manager.get_vectorizer() is not None,
                'anomaly_model_loaded': model_manager.get_anomaly_model() is not None,
                'pattern_model_loaded': model_manager.get_pattern_model() is not None,
                'word2vec_model_loaded': model_manager.get_word2vec_model() is not None
            }
        }

        return jsonify(status_info), 200

    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_logs', methods=['POST'])
@validate_request(['logs'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def analyze_logs():
    """
    Analyze logs for anomalies and patterns.

    Expected JSON body:
    {
        "logs": ["timestamp level message", ...]
    }

    Returns:
    {
        "common_patterns": [...],
        "significant_outliers": [...],
        "visualization": "base64_encoded_image",
        "statistics": {...}
    }
    """
    try:
        # Validate input
        new_logs = request.json['logs']

        if not isinstance(new_logs, list):
            return jsonify({'error': 'logs must be a list'}), 400

        if len(new_logs) == 0:
            return jsonify({'error': 'logs list cannot be empty'}), 400

        if len(new_logs) > config.MAX_BATCH_SIZE:
            return jsonify({
                'error': f'Batch size exceeds maximum of {config.MAX_BATCH_SIZE}'
            }), 400

        logger.info(f"Analyzing {len(new_logs)} logs")

        # Process incoming logs - FIX: avoid double preprocessing
        processed_logs = []
        invalid_count = 0

        for log in new_logs:
            processed = preprocess_log(log)
            if processed is not None:
                processed_logs.append(processed)
            else:
                invalid_count += 1

        if not processed_logs:
            return jsonify({'error': 'No valid logs to analyze'}), 400

        logger.info(f"Processed {len(processed_logs)} valid logs, {invalid_count} invalid")

        # Add to store
        log_store.add_logs(processed_logs)

        # Get all logs for analysis
        all_logs = log_store.get_logs()
        df = pd.DataFrame(all_logs)

        # Feature Extraction
        features = extract_features(df['message'].tolist())

        # Anomaly Detection - FIX: unpack tuple correctly
        anomaly_predictions, anomaly_scores = detect_anomalies(features)

        # Pattern Recognition
        patterns = find_patterns(features)

        # Find most common patterns
        pattern_counter = Counter(patterns)
        common_patterns = pattern_counter.most_common(5)

        # Identify significant outliers (top 5%)
        percentile_95 = np.percentile(anomaly_scores, 95)
        significant_outliers = np.nonzero(anomaly_scores > percentile_95)[0]

        # Generate Visualizations
        viz = generate_visualizations(anomaly_scores, patterns, df['timestamp'])

        # Prepare results
        results = {
            'common_patterns': [
                {
                    'pattern_id': int(pattern),
                    'count': count,
                    'example_logs': df[patterns == pattern]['message'].tolist()[:3]
                } for pattern, count in common_patterns if pattern != -1  # Exclude noise
            ],
            'significant_outliers': [
                {
                    'log_id': int(i),
                    'timestamp': df.iloc[i]['timestamp'].isoformat(),
                    'level': df.iloc[i]['level'],
                    'message': df.iloc[i]['message'],
                    'anomaly_score': float(anomaly_scores[i])
                } for i in significant_outliers[:20]  # Limit to top 20
            ],
            'statistics': {
                'total_logs_analyzed': len(df),
                'new_logs_processed': len(processed_logs),
                'invalid_logs': invalid_count,
                'anomalies_detected': int(np.sum(anomaly_predictions == 1)),
                'patterns_found': len(set(patterns)) - (1 if -1 in patterns else 0),
                'noise_points': int(list(patterns).count(-1))
            },
            'visualization': viz
        }

        logger.info(f"Analysis complete: {results['statistics']}")
        return jsonify(results), 200

    except ValueError as e:
        logger.warning(f"Validation error in analyze_logs: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Error in analyze_logs: {e}", exc_info=True)
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500


@app.route('/train', methods=['POST'])
@validate_request(['logs'])
@limiter.limit(f"{config.RATE_LIMIT_PER_MINUTE}/minute")
@track_time
def train():
    """
    Train models on provided logs.

    Expected JSON body:
    {
        "logs": ["timestamp level message", ...]
    }

    Returns:
    {
        "status": "success",
        "logs_processed": int,
        "models_saved": bool
    }
    """
    try:
        # Validate input
        new_logs = request.json['logs']

        if not isinstance(new_logs, list):
            return jsonify({'error': 'logs must be a list'}), 400

        if len(new_logs) == 0:
            return jsonify({'error': 'logs list cannot be empty'}), 400

        if len(new_logs) > config.MAX_BATCH_SIZE:
            return jsonify({
                'error': f'Batch size exceeds maximum of {config.MAX_BATCH_SIZE}'
            }), 400

        logger.info(f"Training on {len(new_logs)} logs")

        # Process incoming logs
        processed_logs = []
        invalid_count = 0

        for log in new_logs:
            processed = preprocess_log(log)
            if processed is not None:
                processed_logs.append(processed)
            else:
                invalid_count += 1

        if not processed_logs:
            return jsonify({'error': 'No valid logs to train on'}), 400

        logger.info(f"Processed {len(processed_logs)} valid logs, {invalid_count} invalid")

        # Add to store
        log_store.add_logs(processed_logs)

        # Get all logs for training
        all_logs = log_store.get_logs()
        df = pd.DataFrame(all_logs)

        # Feature Extraction
        features = extract_features(df['message'].tolist())

        # Train models - FIX: handle return values correctly
        _, _ = detect_anomalies(features)
        _ = find_patterns(features)
        _ = nlp_analysis(df['message'].tolist())

        # Save models if configured
        models_saved = False
        if config.AUTO_SAVE_MODELS:
            save_models()
            models_saved = True

        result = {
            'status': 'success',
            'logs_processed': len(processed_logs),
            'invalid_logs': invalid_count,
            'total_logs_in_store': len(all_logs),
            'models_saved': models_saved
        }

        logger.info(f"Training complete: {result}")
        return jsonify(result), 200

    except ValueError as e:
        logger.warning(f"Validation error in train: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        logger.error(f"Error in train: {e}", exc_info=True)
        return jsonify({'error': 'Training failed', 'details': str(e)}), 500


@app.route('/clear_logs', methods=['POST'])
@limiter.limit("10/minute")
def clear_logs():
    """Clear all logs from memory."""
    try:
        log_store.clear()
        logger.info("Cleared all logs from store")
        return jsonify({'status': 'success', 'message': 'All logs cleared'}), 200

    except Exception as e:
        logger.error(f"Error clearing logs: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/cleanup', methods=['POST'])
@limiter.limit("10/minute")
def cleanup():
    """Clean up old logs based on retention policy."""
    try:
        removed = log_store.cleanup_old_logs()
        logger.info(f"Cleanup removed {removed} old logs")
        return jsonify({
            'status': 'success',
            'logs_removed': removed,
            'logs_remaining': log_store.size()
        }), 200

    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# Model persistence functions

def save_models() -> None:
    """Save all models to disk."""
    try:
        model_dir = config.MODEL_DIR
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            logger.info(f"Created model directory: {model_dir}")

        # Define model paths
        vectorizer_path = os.path.join(model_dir, f'vectorizer_{config.MODEL_VERSION}.joblib')
        anomaly_path = os.path.join(model_dir, f'anomaly_model_{config.MODEL_VERSION}.joblib')
        pattern_path = os.path.join(model_dir, f'pattern_model_{config.MODEL_VERSION}.joblib')
        word2vec_path = os.path.join(model_dir, f'word2vec_model_{config.MODEL_VERSION}')

        # Save models
        vectorizer = model_manager.get_vectorizer()
        if vectorizer is not None:
            joblib.dump(vectorizer, vectorizer_path)
            logger.info(f"Saved vectorizer to {vectorizer_path}")

        anomaly_model = model_manager.get_anomaly_model()
        if anomaly_model is not None:
            joblib.dump(anomaly_model, anomaly_path)
            logger.info(f"Saved anomaly model to {anomaly_path}")

        pattern_model = model_manager.get_pattern_model()
        if pattern_model is not None:
            joblib.dump(pattern_model, pattern_path)
            logger.info(f"Saved pattern model to {pattern_path}")

        word2vec_model = model_manager.get_word2vec_model()
        if word2vec_model is not None:
            word2vec_model.save(word2vec_path)
            logger.info(f"Saved word2vec model to {word2vec_path}")

        logger.info("All models saved successfully")

    except Exception as e:
        logger.error(f"Error saving models: {e}", exc_info=True)
        raise


def load_models() -> None:
    """Load models from disk."""
    try:
        model_dir = config.MODEL_DIR

        if not os.path.exists(model_dir):
            logger.info(f"Model directory {model_dir} does not exist, skipping model loading")
            return

        # Define model paths
        vectorizer_path = os.path.join(model_dir, f'vectorizer_{config.MODEL_VERSION}.joblib')
        anomaly_path = os.path.join(model_dir, f'anomaly_model_{config.MODEL_VERSION}.joblib')
        pattern_path = os.path.join(model_dir, f'pattern_model_{config.MODEL_VERSION}.joblib')
        word2vec_path = os.path.join(model_dir, f'word2vec_model_{config.MODEL_VERSION}')

        # Load models
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            model_manager.set_vectorizer(vectorizer)
            logger.info(f"Loaded vectorizer from {vectorizer_path}")

        if os.path.exists(anomaly_path):
            anomaly_model = joblib.load(anomaly_path)
            model_manager.set_anomaly_model(anomaly_model)
            logger.info(f"Loaded anomaly model from {anomaly_path}")

        if os.path.exists(pattern_path):
            pattern_model = joblib.load(pattern_path)
            model_manager.set_pattern_model(pattern_model)
            logger.info(f"Loaded pattern model from {pattern_path}")

        if os.path.exists(word2vec_path):
            word2vec_model = Word2Vec.load(word2vec_path)
            model_manager.set_word2vec_model(word2vec_model)
            logger.info(f"Loaded word2vec model from {word2vec_path}")

        logger.info("Model loading complete")

    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        # Don't raise - allow app to start with fresh models


def initialize() -> None:
    """Initialize the application."""
    try:
        logger.info("Initializing Log Analysis API")
        logger.info(f"Configuration: {config.__dict__}")

        # Load existing models
        load_models()

        # Perform initial cleanup
        log_store.cleanup_old_logs()

        logger.info("Initialization complete")

    except Exception as e:
        logger.error(f"Error during initialization: {e}", exc_info=True)
        raise


# Initialize on startup
with app.app_context():
    initialize()


if __name__ == '__main__':
    # Production deployment should use gunicorn instead
    logger.warning("Running in development mode. Use gunicorn for production.")
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG
    )
