# Log Analysis Tool

This application is a powerful log analysis tool that uses machine learning and natural language processing techniques to identify patterns, anomalies, and significant outliers in log data.

## Features

- Log ingestion and preprocessing
- Feature extraction using TF-IDF
- Anomaly detection using Isolation Forest
- Pattern recognition using DBSCAN clustering
- NLP analysis using Word2Vec
- Visualization of log analysis results
- RESTful API for training and analysis

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
