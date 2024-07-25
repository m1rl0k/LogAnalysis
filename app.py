from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from pyod.models.iforest import IForest
from collections import Counter
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
import io
import base64
import joblib
import os

nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Global variables for storing data and models
log_data = []
vectorizer = None
anomaly_model = None
pattern_model = None
word2vec_model = None

def preprocess_log(log):
    parts = log.split(' ', 2)
    if len(parts) < 3:
        return None
    timestamp, level, message = parts
    return {
        'timestamp': pd.to_datetime(timestamp),
        'level': level,
        'message': message
    }

def extract_features(messages):
    global vectorizer
    if vectorizer is None:
        vectorizer = TfidfVectorizer(max_features=100)
        return vectorizer.fit_transform(messages)
    return vectorizer.transform(messages)

def detect_anomalies(features):
    global anomaly_model
    if anomaly_model is None:
        anomaly_model = IForest(contamination=0.1, random_state=42)
    
    # Convert sparse matrix to dense array if necessary
    if scipy.sparse.issparse(features):
        features = features.toarray()
    
    anomaly_model.fit(features)
    return anomaly_model.predict(features)

def find_patterns(features):
    global pattern_model
    if pattern_model is None:
        pattern_model = DBSCAN(eps=0.5, min_samples=2)
    
    # Convert sparse matrix to dense array if necessary
    if scipy.sparse.issparse(features):
        features = features.toarray()
    
    return pattern_model.fit_predict(features)

def nlp_analysis(messages):
    global word2vec_model
    if word2vec_model is None:
        # Preprocess messages to remove log levels and timestamps
        preprocessed_messages = [' '.join(msg.split()[3:]) for msg in messages]
        word2vec_model = Word2Vec(sentences=[simple_preprocess(msg) for msg in preprocessed_messages], vector_size=100, window=5, min_count=1, workers=4)
    
    def get_message_vector(msg):
        # Remove timestamp, log level, and any other metadata
        words = msg.split()[3:]
        word_vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
        if word_vectors:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(word2vec_model.vector_size)

    return [get_message_vector(msg) for msg in messages]

def generate_visualizations(anomalies, patterns, timestamps):
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(timestamps, anomalies, c=patterns, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Log Analysis Visualization')
    plt.xlabel('Timestamp')
    plt.ylabel('Anomaly Score')
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/analyze_logs', methods=['POST'])
def analyze_logs():
    global log_data

    # Process incoming logs
    new_logs = request.json['logs']
    processed_logs = [preprocess_log(log) for log in new_logs if preprocess_log(log) is not None]
    log_data.extend(processed_logs)

    # Convert to DataFrame
    df = pd.DataFrame(log_data)

    # Feature Extraction
    features = extract_features(df['message'])
    if scipy.sparse.issparse(features):
        features = features.toarray()

    # Anomaly Detection
    anomalies = detect_anomalies(features)

    # Pattern Recognition
    patterns = find_patterns(features)

    # NLP Analysis
    nlp_features = nlp_analysis(df['message'])

    # Find most common patterns
    pattern_counter = Counter(patterns)
    common_patterns = pattern_counter.most_common(5)

    # Identify significant outliers
    anomaly_scores = anomaly_model.decision_function(features)
    significant_outliers = np.where(anomaly_scores > np.percentile(anomaly_scores, 95))[0]

    # Generate Visualizations
    viz = generate_visualizations(anomaly_scores, patterns, df['timestamp'])

    # Prepare results
    results = {
        'common_patterns': [
            {
                'pattern_id': int(pattern),
                'count': count,
                'example_logs': df[patterns == pattern]['message'].tolist()[:3]
            } for pattern, count in common_patterns
        ],
        'significant_outliers': [
            {
                'log_id': int(i),
                'message': df.iloc[i]['message'],
                'anomaly_score': float(anomaly_scores[i])
            } for i in significant_outliers
        ],
        'visualization': viz
    }

    return jsonify(results)

@app.route('/train', methods=['POST'])
def train():
    global log_data, vectorizer, anomaly_model, pattern_model, word2vec_model

    new_logs = request.json['logs']
    processed_logs = [preprocess_log(log) for log in new_logs if preprocess_log(log) is not None]
    log_data.extend(processed_logs)

    df = pd.DataFrame(log_data)

    # Feature Extraction
    features = extract_features(df['message'])
    if scipy.sparse.issparse(features):
        features = features.toarray()

    # Anomaly Detection
    detect_anomalies(features)

    # Pattern Recognition
    find_patterns(features)

    # NLP Analysis
    nlp_analysis(df['message'])

    save_models()

    return jsonify({'status': 'Training completed', 'logs_processed': len(new_logs)})

def save_models():
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(vectorizer, 'models/vectorizer.joblib')
    joblib.dump(anomaly_model, 'models/anomaly_model.joblib')
    joblib.dump(pattern_model, 'models/pattern_model.joblib')
    word2vec_model.save('models/word2vec_model')

def load_models():
    global vectorizer, anomaly_model, pattern_model, word2vec_model
    if os.path.exists('models/vectorizer.joblib'):
        vectorizer = joblib.load('models/vectorizer.joblib')
    if os.path.exists('models/anomaly_model.joblib'):
        anomaly_model = joblib.load('models/anomaly_model.joblib')
    if os.path.exists('models/pattern_model.joblib'):
        pattern_model = joblib.load('models/pattern_model.joblib')
    if os.path.exists('models/word2vec_model'):
        word2vec_model = Word2Vec.load('models/word2vec_model')

def initialize():
    load_models()

with app.app_context():
    initialize()

if __name__ == '__main__':
    app.run(debug=True)
