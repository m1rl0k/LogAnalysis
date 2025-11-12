# Log Analysis API Documentation

## Overview

Production-ready REST API for log analysis using machine learning. Provides anomaly detection, pattern recognition, and NLP-based analysis of log data.

**Version:** v1  
**Base URL:** `http://localhost:5000`

---

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding API key authentication or OAuth2.

---

## Rate Limiting

- **Per Minute:** 60 requests
- **Per Hour:** 1000 requests

Rate limits can be configured via environment variables.

---

## Endpoints

### 1. Health Check

Check if the API is running and healthy.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "version": "v1",
  "log_count": 5000,
  "models_loaded": {
    "vectorizer": true,
    "anomaly_model": true,
    "pattern_model": true,
    "word2vec_model": true
  }
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `500 Internal Server Error` - Service is unhealthy

---

### 2. System Status

Get detailed system status and statistics.

**Endpoint:** `GET /status`

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "version": "v1",
  "configuration": {
    "max_logs_in_memory": 100000,
    "max_batch_size": 10000,
    "log_retention_hours": 24,
    "rate_limit_enabled": true
  },
  "statistics": {
    "total_logs": 5000,
    "memory_usage_percent": 5.0
  },
  "models": {
    "vectorizer_loaded": true,
    "anomaly_model_loaded": true,
    "pattern_model_loaded": true,
    "word2vec_model_loaded": true
  }
}
```

**Status Codes:**
- `200 OK` - Success
- `500 Internal Server Error` - Error retrieving status

---

### 3. Analyze Logs

Analyze logs for anomalies and patterns.

**Endpoint:** `POST /analyze_logs`

**Request Body:**
```json
{
  "logs": [
    "2024-01-15T10:30:00 INFO Application started successfully",
    "2024-01-15T10:30:01 ERROR Database connection failed",
    "2024-01-15T10:30:02 WARNING High memory usage detected"
  ]
}
```

**Request Parameters:**
- `logs` (array, required): Array of log strings in format "timestamp level message"

**Response:**
```json
{
  "common_patterns": [
    {
      "pattern_id": 0,
      "count": 150,
      "example_logs": [
        "Application started successfully",
        "Service initialized",
        "Ready to accept connections"
      ]
    }
  ],
  "significant_outliers": [
    {
      "log_id": 42,
      "timestamp": "2024-01-15T10:30:01",
      "level": "ERROR",
      "message": "Database connection failed",
      "anomaly_score": 0.85
    }
  ],
  "statistics": {
    "total_logs_analyzed": 5000,
    "new_logs_processed": 3,
    "invalid_logs": 0,
    "anomalies_detected": 25,
    "patterns_found": 5,
    "noise_points": 10
  },
  "visualization": "base64_encoded_png_image"
}
```

**Status Codes:**
- `200 OK` - Analysis successful
- `400 Bad Request` - Invalid input
- `413 Payload Too Large` - Batch size exceeds limit
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Analysis failed

---

### 4. Train Models

Train ML models on provided logs.

**Endpoint:** `POST /train`

**Request Body:**
```json
{
  "logs": [
    "2024-01-15T10:30:00 INFO Application started",
    "2024-01-15T10:30:01 ERROR Connection timeout"
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "logs_processed": 2,
  "invalid_logs": 0,
  "total_logs_in_store": 5002,
  "models_saved": true
}
```

**Status Codes:**
- `200 OK` - Training successful
- `400 Bad Request` - Invalid input
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Training failed

---

### 5. Clear Logs

Clear all logs from memory.

**Endpoint:** `POST /clear_logs`

**Response:**
```json
{
  "status": "success",
  "message": "All logs cleared"
}
```

**Status Codes:**
- `200 OK` - Logs cleared
- `500 Internal Server Error` - Clear failed

---

### 6. Cleanup Old Logs

Remove logs older than retention period.

**Endpoint:** `POST /cleanup`

**Response:**
```json
{
  "status": "success",
  "logs_removed": 150,
  "logs_remaining": 4850
}
```

**Status Codes:**
- `200 OK` - Cleanup successful
- `500 Internal Server Error` - Cleanup failed

---

### 7. Signal Preprocessing

Apply advanced signal processing filters to numeric data columns with comprehensive visualization.

**Endpoint:** `POST /preprocess_signal`

**Request Body:**
```json
{
  "dataset": "sensor_data",
  "columns": ["temperature", "pressure"],
  "signal_type": "auto",
  "visualize": true,
  "apply_to_dataset": false
}
```

**Parameters:**
- `dataset` (required): Name of the dataset to process
- `columns` (optional): Array of column names to process. Defaults to all numeric columns
- `signal_type` (optional): Signal type hint. Default: "auto"
- `visualize` (optional): Generate visualizations. Default: true
- `apply_to_dataset` (optional): Update dataset with cleaned signals. Default: false

**Response:**
```json
{
  "dataset": "sensor_data",
  "columns_processed": ["temperature", "pressure"],
  "results": {
    "temperature": {
      "method": "BUTTERWORTH+KALMAN",
      "snr_improvement_db": 18.5,
      "all_methods_snr": {
        "butterworth": 17.2,
        "savgol": 15.8,
        "kalman": 19.3,
        "hampel": 14.1,
        "wavelet": 16.7
      },
      "original_length": 1000,
      "cleaned_length": 1000,
      "noise_std": 0.234,
      "signal_std_original": 2.456,
      "signal_std_cleaned": 2.389
    }
  },
  "applied_to_dataset": false,
  "visualizations": {
    "temperature": {
      "base64": "iVBORw0KGgoAAAANSUhEUgAA...",
      "path": "analysis/sensor_data_temperature_signal_processing_20240115_103000.png"
    }
  }
}
```

**Visualization Features:**
- Original vs Filtered signal comparison
- FFT frequency analysis (before/after)
- SNR comparison across all methods
- Noise removed visualization
- Statistical summaries

**Supported Filters:**
- Butterworth low-pass filter (auto-tuned cutoff)
- Savitzky-Golay filter (peak-preserving)
- Kalman filter (optimal state estimation)
- Hampel filter (outlier removal)
- Wavelet denoising (multi-resolution)

**Status Codes:**
- `200 OK` - Processing successful
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Dataset not found
- `500 Internal Server Error` - Processing failed

---

### 8. Comprehensive Analysis (with Signal Processing)

Run complete analysis including signal preprocessing, outlier detection, anomaly detection, and forecasting.

**Endpoint:** `POST /analyze`

**Request Body:**
```json
{
  "dataset": "sensor_data",
  "target": "temperature",
  "preprocess_signals": true,
  "signal_columns": ["temperature", "pressure"]
}
```

**Parameters:**
- `dataset` or `file` (required): Dataset name or file path
- `target` (optional): Target column for regression
- `preprocess_signals` (optional): Enable signal preprocessing. Default: false
- `signal_columns` (optional): Columns to preprocess. Defaults to first 3 numeric columns

**Response:**
```json
{
  "dataset": "sensor_data",
  "summary": {
    "rows": 1000,
    "columns": 5,
    "numeric_columns": 3
  },
  "signal_processing": {
    "columns_processed": ["temperature", "pressure"],
    "results": {
      "temperature": {
        "method": "BUTTERWORTH+KALMAN",
        "snr_improvement_db": 18.5,
        "all_methods_snr": {...}
      }
    },
    "visualizations": {...}
  },
  "outliers": {...},
  "anomalies": {...},
  "forecasts": {...},
  "visualization": "base64_encoded_image",
  "visualization_path": "analysis/sensor_data_analysis_20240115_103000.png"
}
```

**Status Codes:**
- `200 OK` - Analysis successful
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Dataset/file not found
- `500 Internal Server Error` - Analysis failed

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "details": "Additional context (optional)"
}
```

---

## Log Format

Logs should follow this format:
```
TIMESTAMP LEVEL MESSAGE
```

**Examples:**
- `2024-01-15T10:30:00 INFO Application started`
- `2024-01-15T10:30:01.123Z ERROR Database connection failed`
- `2024-01-15 10:30:02 WARNING High CPU usage`

**Supported Log Levels:**
- DEBUG
- INFO
- WARNING / WARN
- ERROR
- CRITICAL / FATAL

---

## Metrics (Prometheus)

Prometheus metrics are available at `http://localhost:9090/metrics` when `ENABLE_METRICS=True`.

**Available Metrics:**
- Request count
- Request duration
- Response status codes
- Active requests
- Application info

---

## Best Practices

1. **Batch Processing:** Send logs in batches (up to 10,000) for better performance
2. **Regular Cleanup:** Call `/cleanup` periodically to manage memory
3. **Model Training:** Train models with representative data before analysis
4. **Error Handling:** Always check response status codes
5. **Rate Limiting:** Implement exponential backoff for rate limit errors

---

## Examples

### Python Example

```python
import requests

# Analyze logs
response = requests.post(
    'http://localhost:5000/analyze_logs',
    json={
        'logs': [
            '2024-01-15T10:30:00 INFO App started',
            '2024-01-15T10:30:01 ERROR Connection failed'
        ]
    }
)

if response.status_code == 200:
    results = response.json()
    print(f"Found {len(results['significant_outliers'])} anomalies")
else:
    print(f"Error: {response.json()}")
```

### cURL Example

```bash
curl -X POST http://localhost:5000/analyze_logs \
  -H "Content-Type: application/json" \
  -d '{
    "logs": [
      "2024-01-15T10:30:00 INFO Application started",
      "2024-01-15T10:30:01 ERROR Database error"
    ]
  }'
```

