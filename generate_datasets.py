#!/usr/bin/env python3
"""
Generate large, realistic datasets for testing the Analytics API.
Creates CSV, Excel, and JSON formats for various data types.
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

# Create data directory
os.makedirs('data', exist_ok=True)

print("Generating large realistic datasets...")

# ==================== 1. SERVER LOGS (10,000 entries) ====================
print("\n1. Generating server logs (10,000 entries)...")
np.random.seed(42)

start_date = datetime(2024, 1, 1)
log_data = []

for i in range(10000):
    timestamp = start_date + timedelta(minutes=i)
    status_codes = [200, 200, 200, 200, 200, 304, 404, 500, 503]  # Weighted
    response_time = np.random.lognormal(4, 1.5)  # Log-normal distribution
    bytes_sent = np.random.randint(100, 50000)
    
    # Add anomalies
    if i % 500 == 0:  # Periodic spikes
        response_time *= 10
    if i % 1000 == 0:  # Occasional errors
        status_codes = [500, 503]
    
    log_data.append({
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'status_code': np.random.choice(status_codes),
        'response_time_ms': round(response_time, 2),
        'bytes_sent': bytes_sent,
        'endpoint': np.random.choice(['/api/users', '/api/products', '/api/orders', '/health', '/metrics']),
        'method': np.random.choice(['GET', 'POST', 'PUT', 'DELETE'], p=[0.7, 0.2, 0.05, 0.05])
    })

logs_df = pd.DataFrame(log_data)
logs_df.to_csv('data/server_logs.csv', index=False)
logs_df.to_excel('data/server_logs.xlsx', index=False)
logs_df.to_json('data/server_logs.json', orient='records', indent=2)
print(f"   ✓ Created server_logs: {len(logs_df)} rows")

# ==================== 2. SALES DATA (5,000 entries) ====================
print("\n2. Generating sales data (5,000 entries)...")

sales_data = []
for i in range(5000):
    date = start_date + timedelta(days=i % 365)
    day_of_week = date.weekday()
    month = date.month
    
    # Seasonal patterns
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month / 12)
    weekend_boost = 1.5 if day_of_week >= 5 else 1.0
    
    base_sales = 1000
    marketing = np.random.randint(50, 500)
    temperature = 15 + 10 * np.sin(2 * np.pi * month / 12) + np.random.normal(0, 3)
    
    sales = base_sales * seasonal_factor * weekend_boost + marketing * 2 + temperature * 5
    sales += np.random.normal(0, 100)
    
    # Add outliers
    if np.random.random() < 0.02:
        sales *= np.random.choice([0.3, 2.5])
    
    sales_data.append({
        'date': date.strftime('%Y-%m-%d'),
        'sales': round(sales, 2),
        'marketing_spend': marketing,
        'temperature': round(temperature, 1),
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': day_of_week >= 5,
        'season': ['Winter', 'Winter', 'Spring', 'Spring', 'Spring', 'Summer', 
                   'Summer', 'Summer', 'Fall', 'Fall', 'Fall', 'Winter'][month-1]
    })

sales_df = pd.DataFrame(sales_data)
sales_df.to_csv('data/sales_data.csv', index=False)
sales_df.to_excel('data/sales_data.xlsx', index=False)
sales_df.to_json('data/sales_data.json', orient='records', indent=2)
print(f"   ✓ Created sales_data: {len(sales_df)} rows")

# ==================== 3. SENSOR/IoT DATA (20,000 entries) ====================
print("\n3. Generating sensor/IoT data (20,000 entries)...")

sensor_data = []
sensors = ['SENSOR_A1', 'SENSOR_A2', 'SENSOR_B1', 'SENSOR_B2', 'SENSOR_C1']

for i in range(20000):
    timestamp = start_date + timedelta(minutes=i)
    sensor_id = sensors[i % len(sensors)]
    
    # Normal operating ranges
    temp = 22 + np.random.normal(0, 2)
    humidity = 45 + np.random.normal(0, 5)
    pressure = 1013 + np.random.normal(0, 3)
    vibration = 0.5 + np.random.normal(0, 0.1)
    
    # Add anomalies (sensor failures, spikes)
    if np.random.random() < 0.005:
        temp += np.random.choice([-20, 50])
    if np.random.random() < 0.005:
        vibration *= 10
    
    sensor_data.append({
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'sensor_id': sensor_id,
        'temperature': round(temp, 2),
        'humidity': round(humidity, 1),
        'pressure': round(pressure, 1),
        'vibration': round(vibration, 3),
        'status': 'OK' if -10 < temp < 50 and vibration < 2 else 'ALERT'
    })

sensor_df = pd.DataFrame(sensor_data)
sensor_df.to_csv('data/sensor_data.csv', index=False)
sensor_df.to_excel('data/sensor_data.xlsx', index=False)
sensor_df.to_json('data/sensor_data.json', orient='records', indent=2)
print(f"   ✓ Created sensor_data: {len(sensor_df)} rows")

# ==================== 4. WEATHER DATA (3,650 entries - 10 years) ====================
print("\n4. Generating weather data (3,650 entries - 10 years)...")

weather_data = []
for i in range(3650):
    date = start_date + timedelta(days=i)
    month = date.month

    # Seasonal temperature patterns
    base_temp = 15 + 15 * np.sin(2 * np.pi * (month - 3) / 12)
    temp = base_temp + np.random.normal(0, 5)

    # Humidity inversely related to temperature
    humidity = 70 - (temp - 15) * 1.5 + np.random.normal(0, 10)
    humidity = max(20, min(100, humidity))

    # Precipitation
    precip_prob = 0.3 if month in [11, 12, 1, 2, 3] else 0.15
    precipitation = np.random.exponential(5) if np.random.random() < precip_prob else 0

    # Wind speed
    wind_speed = abs(np.random.normal(15, 8))

    # Pressure
    pressure = 1013 + np.random.normal(0, 10)

    weather_data.append({
        'date': date.strftime('%Y-%m-%d'),
        'temperature': round(temp, 1),
        'humidity': round(humidity, 1),
        'precipitation_mm': round(precipitation, 1),
        'wind_speed_kmh': round(wind_speed, 1),
        'pressure_hpa': round(pressure, 1),
        'cloud_cover': np.random.randint(0, 101),
        'condition': np.random.choice(['Sunny', 'Cloudy', 'Rainy', 'Stormy'],
                                     p=[0.4, 0.3, 0.2, 0.1])
    })

weather_df = pd.DataFrame(weather_data)
weather_df.to_csv('data/weather_data.csv', index=False)
weather_df.to_excel('data/weather_data.xlsx', index=False)
weather_df.to_json('data/weather_data.json', orient='records', indent=2)
print(f"   ✓ Created weather_data: {len(weather_df)} rows")

# ==================== 5. STOCK PRICES (2,000 trading days) ====================
print("\n5. Generating stock price data (2,000 trading days)...")

stock_data = []
price = 100.0

for i in range(2000):
    date = start_date + timedelta(days=i)
    # Skip weekends
    if date.weekday() >= 5:
        continue

    # Random walk with drift
    daily_return = np.random.normal(0.0005, 0.02)
    price *= (1 + daily_return)

    # OHLC data
    open_price = price * (1 + np.random.normal(0, 0.005))
    high = max(open_price, price) * (1 + abs(np.random.normal(0, 0.01)))
    low = min(open_price, price) * (1 - abs(np.random.normal(0, 0.01)))
    close = price

    volume = int(np.random.lognormal(14, 0.5))

    stock_data.append({
        'date': date.strftime('%Y-%m-%d'),
        'open': round(open_price, 2),
        'high': round(high, 2),
        'low': round(low, 2),
        'close': round(close, 2),
        'volume': volume,
        'adj_close': round(close, 2)
    })

stock_df = pd.DataFrame(stock_data)
stock_df.to_csv('data/stock_prices.csv', index=False)
stock_df.to_excel('data/stock_prices.xlsx', index=False)
stock_df.to_json('data/stock_prices.json', orient='records', indent=2)
print(f"   ✓ Created stock_prices: {len(stock_df)} rows")

# ==================== 6. E-COMMERCE TRANSACTIONS (15,000 entries) ====================
print("\n6. Generating e-commerce transaction data (15,000 entries)...")

transaction_data = []
for i in range(15000):
    timestamp = start_date + timedelta(hours=i % (24*365), minutes=np.random.randint(0, 60))

    # Transaction amount with realistic distribution
    amount = np.random.lognormal(4, 1.2)

    # Customer segments
    segment = np.random.choice(['Premium', 'Regular', 'Budget'], p=[0.2, 0.5, 0.3])
    if segment == 'Premium':
        amount *= 2
    elif segment == 'Budget':
        amount *= 0.5

    transaction_data.append({
        'transaction_id': f'TXN{i+1:06d}',
        'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'amount': round(amount, 2),
        'customer_segment': segment,
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Debit Card', 'Crypto'],
                                          p=[0.5, 0.3, 0.15, 0.05]),
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports']),
        'quantity': np.random.randint(1, 6),
        'discount_applied': np.random.choice([True, False], p=[0.3, 0.7]),
        'shipping_cost': round(np.random.uniform(0, 20), 2)
    })

ecommerce_df = pd.DataFrame(transaction_data)
ecommerce_df.to_csv('data/ecommerce_transactions.csv', index=False)
ecommerce_df.to_excel('data/ecommerce_transactions.xlsx', index=False)
ecommerce_df.to_json('data/ecommerce_transactions.json', orient='records', indent=2)
print(f"   ✓ Created ecommerce_transactions: {len(ecommerce_df)} rows")

print("\n" + "="*60)
print("✓ Dataset generation complete!")
print("="*60)
print(f"\nGenerated files in 'data/' directory:")
print(f"  - server_logs: 10,000 rows (CSV, Excel, JSON)")
print(f"  - sales_data: 5,000 rows (CSV, Excel, JSON)")
print(f"  - sensor_data: 20,000 rows (CSV, Excel, JSON)")
print(f"  - weather_data: 3,650 rows (CSV, Excel, JSON)")
print(f"  - stock_prices: ~1,400 rows (CSV, Excel, JSON)")
print(f"  - ecommerce_transactions: 15,000 rows (CSV, Excel, JSON)")
print(f"\nTotal: 6 datasets × 3 formats = 18 files")
print("\nMove old data:")
print("  mv mock_data/* data/ 2>/dev/null || true")
print("  mv sample_data/* data/ 2>/dev/null || true")

