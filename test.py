import argparse
import requests
import json
import base64
from tqdm import tqdm

def train_model(logs, batch_size=1000):
    url = 'http://localhost:5000/train'
    
    for i in tqdm(range(0, len(logs), batch_size)):
        batch = logs[i:i+batch_size]
        response = requests.post(url, json={'logs': batch})
        if response.status_code != 200:
            print(f"Error processing batch {i//batch_size}: {response.text}")
        else:
            print(response.json()['status'])

def analyze_logs(logs):
    url = 'http://localhost:5000/analyze_logs'
    response = requests.post(url, json={'logs': logs})
    if response.status_code == 200:
        results = response.json()
        print("Common Patterns:")
        for pattern in results['common_patterns']:
            print(f"Pattern {pattern['pattern_id']}: {pattern['count']} occurrences")
            print("Example logs:")
            for log in pattern['example_logs']:
                print(f"  - {log}")
            print()
        
        print("Significant Outliers:")
        for outlier in results['significant_outliers']:
            print(f"Log ID {outlier['log_id']}: {outlier['message']}")
            print(f"Anomaly Score: {outlier['anomaly_score']}")
            print()
        
        # Save the visualization as an image file
        with open("log_visualization.png", "wb") as f:
            f.write(base64.b64decode(results['visualization']))
        print("Visualization saved as log_visualization.png")
    else:
        print("Error:", response.text)

def main():
    parser = argparse.ArgumentParser(description="Log Analysis Tool")
    parser.add_argument("-t", "--train", action="store_true", help="Run in training mode")
    args = parser.parse_args()

    # Load your log data
    with open('log_dataset.json', 'r') as f:
        log_data = json.load(f)

    if args.train:
        print("Running in training mode...")
        train_model(log_data)
    else:
        print("Running in analysis mode...")
        analyze_logs(log_data)

if __name__ == "__main__":
    main()
