# Deployment Guide

## Quick Start

### Local Development

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env and set SECRET_KEY and other variables
```

3. **Run the Application**
```bash
# Development mode
python app.py

# Production mode with Gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 4 --threads 2 --timeout 120 app:app
```

4. **Test the API**
```bash
curl http://localhost:5000/health
```

---

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t log-analysis-api .

# Run the container
docker run -d \
  -p 5000:5000 \
  -p 9090:9090 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -e SECRET_KEY=your-secret-key \
  --name log-analysis \
  log-analysis-api
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Production Deployment

### Prerequisites

- Python 3.11+
- 4GB RAM minimum (8GB recommended)
- 2 CPU cores minimum
- 10GB disk space

### Environment Variables

**Required:**
- `SECRET_KEY` - Set to a random string (use `python -c "import os; print(os.urandom(32).hex())"`)

**Recommended:**
- `LOG_LEVEL=INFO`
- `MAX_LOGS_IN_MEMORY=100000`
- `RATE_LIMIT_ENABLED=True`

### Systemd Service

Create `/etc/systemd/system/log-analysis.service`:

```ini
[Unit]
Description=Log Analysis API
After=network.target

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/log-analysis
Environment="PATH=/opt/log-analysis/venv/bin"
EnvironmentFile=/opt/log-analysis/.env
ExecStart=/opt/log-analysis/venv/bin/gunicorn \
    --bind 0.0.0.0:5000 \
    --workers 4 \
    --threads 2 \
    --timeout 120 \
    --access-logfile /var/log/log-analysis/access.log \
    --error-logfile /var/log/log-analysis/error.log \
    app:app

Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable log-analysis
sudo systemctl start log-analysis
sudo systemctl status log-analysis
```

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 120s;
        proxy_send_timeout 120s;
        proxy_read_timeout 120s;
        
        # Buffer settings
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
    }
    
    location /metrics {
        proxy_pass http://127.0.0.1:9090;
        # Restrict access to metrics
        allow 10.0.0.0/8;
        deny all;
    }
}
```

---

## Monitoring

### Health Checks

```bash
# Basic health check
curl http://localhost:5000/health

# Detailed status
curl http://localhost:5000/status
```

### Prometheus Metrics

Access metrics at `http://localhost:9090/metrics`

**Key Metrics:**
- `flask_http_request_total` - Total requests
- `flask_http_request_duration_seconds` - Request duration
- `flask_http_request_exceptions_total` - Exceptions

### Logging

Logs are written to:
- Console (stdout/stderr)
- File (if `LOG_FILE` is set)

**Log Levels:**
- `DEBUG` - Detailed debugging
- `INFO` - General information (default)
- `WARNING` - Warning messages
- `ERROR` - Error messages
- `CRITICAL` - Critical errors

---

## Maintenance

### Backup Models

```bash
# Backup models directory
tar -czf models-backup-$(date +%Y%m%d).tar.gz models/
```

### Clear Old Logs

```bash
# Via API
curl -X POST http://localhost:5000/cleanup

# Manual cleanup
find logs/ -name "*.log" -mtime +7 -delete
```

### Update Application

```bash
# Pull latest code
git pull

# Install dependencies
pip install -r requirements.txt

# Restart service
sudo systemctl restart log-analysis
```

---

## Troubleshooting

### High Memory Usage

1. Reduce `MAX_LOGS_IN_MEMORY`
2. Call `/cleanup` more frequently
3. Reduce `LOG_RETENTION_HOURS`

### Slow Analysis

1. Reduce `MAX_BATCH_SIZE`
2. Increase worker count
3. Add more CPU/RAM

### Rate Limiting Issues

1. Increase `RATE_LIMIT_PER_MINUTE`
2. Disable rate limiting: `RATE_LIMIT_ENABLED=False`

### Model Loading Errors

1. Check model directory permissions
2. Verify model files exist
3. Check logs for detailed errors

---

## Security Checklist

- [ ] Set strong `SECRET_KEY`
- [ ] Enable rate limiting
- [ ] Use HTTPS in production
- [ ] Restrict `/metrics` endpoint
- [ ] Keep dependencies updated
- [ ] Monitor logs for suspicious activity
- [ ] Set appropriate file permissions
- [ ] Use firewall rules
- [ ] Regular backups
- [ ] Implement authentication (if needed)

