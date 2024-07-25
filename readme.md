# Log Analysis Tool

This advanced log analysis tool leverages machine learning and natural language processing techniques to analyze log data, identify patterns, detect anomalies, and extract meaningful insights from large volumes of log entries.

## What It Does

This tool performs several key functions on log data:

1. Pattern Recognition: Identifies common patterns in log messages, helping to group similar events and detect recurring issues.

2. Anomaly Detection: Flags unusual log entries that deviate from the norm, which could indicate potential problems or security threats.

3. NLP Analysis: Applies natural language processing to understand the content of log messages, enabling more intelligent analysis.

4. Temporal Analysis: Examines how log patterns and anomalies change over time, providing insights into system behavior trends.

5. Visualization: Generates visual representations of the analysis results, making it easier to interpret complex log data.

## Key Features

### 1. Log Ingestion and Preprocessing
- Parses various log formats, extracting timestamp, log level, and message content.
- Handles large volumes of log data efficiently.

### 2. Feature Extraction
- Utilizes TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert log messages into numerical features.
- This allows the tool to understand the importance of different words in the context of the entire log dataset.

### 3. Anomaly Detection
- Employs Isolation Forest algorithm to identify outliers in log data.
- Isolation Forest is particularly effective for high-dimensional data and can detect anomalies that might be missed by traditional methods.

### 4. Pattern Recognition
- Uses DBSCAN (Density-Based Spatial Clustering of Applications with Noise) for identifying clusters of similar log entries.
- DBSCAN is capable of discovering clusters of arbitrary shape and is robust to outliers.

### 5. NLP Analysis
- Incorporates Word2Vec model to understand semantic relationships between words in log messages.
- This allows for more nuanced analysis, capturing the meaning behind log entries rather than just literal matches.

### 6. Visualization
- Generates a scatter plot showing:
  - Log entries over time (x-axis)
  - Anomaly scores (y-axis)
  - Detected patterns (color-coded)
- This visualization helps in quickly identifying temporal patterns and anomalies.

### 7. API Interface
- Provides a RESTful API for easy integration with other tools and systems.
- Supports both training of models and analysis of new log data.

### 8. Adaptive Learning
- The tool can be retrained on new log data, allowing it to adapt to evolving system behaviors and log patterns over time.

## Use Cases

- IT Operations: Quickly identify issues in system logs that might indicate performance problems or failures.
- Security Monitoring: Detect unusual patterns in log data that could signify security breaches or attempts.
- Application Debugging: Analyze application logs to find recurring errors or performance bottlenecks.
- Compliance Auditing: Identify logs that deviate from expected patterns, which might indicate compliance issues.

## How It Works

1. Log data is ingested and preprocessed to extract relevant information.
2. The TF-IDF vectorizer converts log messages into numerical features.
3. The Isolation Forest algorithm identifies anomalies based on these features.
4. DBSCAN clustering groups similar log entries to identify common patterns.
5. Word2Vec is used to analyze the semantic content of log messages.
6. Results are compiled, including common patterns, significant outliers, and NLP insights.
7. A visualization is generated to represent the findings graphically.

This tool provides a comprehensive approach to log analysis, combining statistical methods, machine learning, and natural language processing to extract maximum value from log data.

## Installation

1. Create a new directory for the project and copy the following files into it:
   - `app.py`
   - `test.py`
   - `log_dataset.json` (your log data file)

2. Install the required dependencies:
   ```
   pip install flask pandas numpy scipy scikit-learn pyod gensim nltk matplotlib joblib tqdm requests argparse
   ```

3. Download the NLTK 'punkt' tokenizer data:
   ```python
   import nltk
   nltk.download('punkt')
   ```

## Usage

### Starting the Server

Run the Flask application:

```
python app.py
```

The server will start running on `http://localhost:5000`.

### Running the Client

The `test.py` script can be used to train the model or analyze logs.

1. For training:
   ```
   python test.py -t -f path_to_your_logfile.json
   ```

2. For analysis:
   ```
   python test.py -f path_to_your_logfile.json
   ```

If you don't specify a file with `-f`, it will default to `log_dataset.json` in the current directory.

## API Endpoints

- `/train` (POST): Train the model with new log data
- `/analyze_logs` (POST): Analyze log data and return results

## Input Data Format

The input log data should be a JSON array of log entries. Each log entry should be a string that includes a timestamp, log level, and message. For example:

```json
[
  "2023-07-25 10:00:01 INFO User authentication: User 'jsmith' logged in successfully",
  "2023-07-25 10:00:02 DEBUG Database connection established: Connection ID 1234",
  "2023-07-25 10:00:03 ERROR Database connection lost: Timeout after 30 seconds"
]
```

## Output

### Analysis Results

The analysis results include:

1. Common patterns found in the logs
2. Significant outliers detected
3. A visualization of the log analysis

### Visualization

The script generates a visualization saved as `log_visualization.png`. This plot shows:

- X-axis: Timestamps of the log entries
- Y-axis: Anomaly scores
- Color: Different colors represent different patterns identified in the logs

This visualization helps in quickly identifying temporal patterns, anomalies, and their relationships.

### Training Output

When running in training mode, the script will:

1. Process the input logs in batches
2. Train the models (TF-IDF vectorizer, Isolation Forest for anomaly detection, DBSCAN for pattern recognition, and Word2Vec for NLP analysis)
3. Save the trained models in a `models` directory:
   - `vectorizer.joblib`: TF-IDF vectorizer
   - `anomaly_model.joblib`: Isolation Forest model
   - `pattern_model.joblib`: DBSCAN model
   - `word2vec_model`: Word2Vec model

These saved models are then used when running the analysis, ensuring consistent processing across multiple runs.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
