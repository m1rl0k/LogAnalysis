"""
Production-ready configuration management for Log Analysis API.
Supports environment-based configuration with validation and sensible defaults.
"""
import os
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Config:
    """Application configuration with environment variable support."""
    
    # Flask Configuration
    FLASK_ENV: str = field(default_factory=lambda: os.getenv('FLASK_ENV', 'production'))
    DEBUG: bool = field(default_factory=lambda: os.getenv('DEBUG', 'False').lower() == 'true')
    HOST: str = field(default_factory=lambda: os.getenv('HOST', '0.0.0.0'))
    PORT: int = field(default_factory=lambda: int(os.getenv('PORT', '5000')))
    
    # Security
    SECRET_KEY: str = field(default_factory=lambda: os.getenv('SECRET_KEY', os.urandom(32).hex()))
    MAX_CONTENT_LENGTH: int = field(default_factory=lambda: int(os.getenv('MAX_CONTENT_LENGTH', str(16 * 1024 * 1024))))  # 16MB
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = field(default_factory=lambda: os.getenv('RATE_LIMIT_ENABLED', 'True').lower() == 'true')
    RATE_LIMIT_PER_MINUTE: int = field(default_factory=lambda: int(os.getenv('RATE_LIMIT_PER_MINUTE', '60')))
    RATE_LIMIT_PER_HOUR: int = field(default_factory=lambda: int(os.getenv('RATE_LIMIT_PER_HOUR', '1000')))
    
    # Memory Management
    MAX_LOGS_IN_MEMORY: int = field(default_factory=lambda: int(os.getenv('MAX_LOGS_IN_MEMORY', '100000')))
    MAX_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv('MAX_BATCH_SIZE', '10000')))
    LOG_RETENTION_HOURS: int = field(default_factory=lambda: int(os.getenv('LOG_RETENTION_HOURS', '24')))
    
    # ML Model Configuration
    TFIDF_MAX_FEATURES: int = field(default_factory=lambda: int(os.getenv('TFIDF_MAX_FEATURES', '100')))
    ANOMALY_CONTAMINATION: float = field(default_factory=lambda: float(os.getenv('ANOMALY_CONTAMINATION', '0.1')))
    DBSCAN_EPS: float = field(default_factory=lambda: float(os.getenv('DBSCAN_EPS', '0.5')))
    DBSCAN_MIN_SAMPLES: int = field(default_factory=lambda: int(os.getenv('DBSCAN_MIN_SAMPLES', '2')))
    WORD2VEC_VECTOR_SIZE: int = field(default_factory=lambda: int(os.getenv('WORD2VEC_VECTOR_SIZE', '100')))
    WORD2VEC_WINDOW: int = field(default_factory=lambda: int(os.getenv('WORD2VEC_WINDOW', '5')))
    WORD2VEC_MIN_COUNT: int = field(default_factory=lambda: int(os.getenv('WORD2VEC_MIN_COUNT', '1')))
    IFOREST_RANDOM_STATE: int = field(default_factory=lambda: int(os.getenv('IFOREST_RANDOM_STATE', '42')))
    
    # Model Persistence
    MODEL_DIR: str = field(default_factory=lambda: os.getenv('MODEL_DIR', 'models'))
    AUTO_SAVE_MODELS: bool = field(default_factory=lambda: os.getenv('AUTO_SAVE_MODELS', 'True').lower() == 'true')
    MODEL_VERSION: str = field(default_factory=lambda: os.getenv('MODEL_VERSION', 'v1'))
    
    # Logging Configuration
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    LOG_FORMAT: str = field(default_factory=lambda: os.getenv(
        'LOG_FORMAT',
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    LOG_FILE: Optional[str] = field(default_factory=lambda: os.getenv('LOG_FILE'))
    
    # Monitoring
    ENABLE_METRICS: bool = field(default_factory=lambda: os.getenv('ENABLE_METRICS', 'True').lower() == 'true')
    METRICS_PORT: int = field(default_factory=lambda: int(os.getenv('METRICS_PORT', '9090')))
    
    # Visualization
    VISUALIZATION_DPI: int = field(default_factory=lambda: int(os.getenv('VISUALIZATION_DPI', '100')))
    VISUALIZATION_FORMAT: str = field(default_factory=lambda: os.getenv('VISUALIZATION_FORMAT', 'png'))
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.MAX_LOGS_IN_MEMORY < 1000:
            raise ValueError("MAX_LOGS_IN_MEMORY must be at least 1000")
        
        if self.MAX_BATCH_SIZE < 1 or self.MAX_BATCH_SIZE > self.MAX_LOGS_IN_MEMORY:
            raise ValueError(f"MAX_BATCH_SIZE must be between 1 and {self.MAX_LOGS_IN_MEMORY}")
        
        if not 0 < self.ANOMALY_CONTAMINATION < 0.5:
            raise ValueError("ANOMALY_CONTAMINATION must be between 0 and 0.5")
        
        if self.DBSCAN_EPS <= 0:
            raise ValueError("DBSCAN_EPS must be positive")
        
        if self.DBSCAN_MIN_SAMPLES < 1:
            raise ValueError("DBSCAN_MIN_SAMPLES must be at least 1")
        
        if self.LOG_LEVEL not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
            raise ValueError(f"Invalid LOG_LEVEL: {self.LOG_LEVEL}")
        
        if self.RATE_LIMIT_PER_MINUTE < 1:
            raise ValueError("RATE_LIMIT_PER_MINUTE must be at least 1")
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()


# Global configuration instance
config = Config()

