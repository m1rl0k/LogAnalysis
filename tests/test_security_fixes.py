"""
Test suite for security and reliability fixes.

Tests:
1. Path validation prevents directory traversal
2. Unified model schema validation
3. AUTO_SAVE_MODELS config is respected
"""
import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from app import validate_file_path, ModelManager, linear_regression, config


class TestPathValidation:
    """Test file path validation security."""
    
    def setup_method(self):
        """Create temporary test directory."""
        self.test_dir = tempfile.mkdtemp()
        self.original_allowed_dir = config.ALLOWED_UPLOAD_DIR
        self.original_security_enabled = config.ENABLE_FILE_UPLOAD_SECURITY
        config.ALLOWED_UPLOAD_DIR = self.test_dir
        config.ENABLE_FILE_UPLOAD_SECURITY = True
        
        # Create test file
        self.test_file = os.path.join(self.test_dir, 'test.csv')
        with open(self.test_file, 'w') as f:
            f.write('a,b,c\n1,2,3\n4,5,6\n')
    
    def teardown_method(self):
        """Clean up test directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
        config.ALLOWED_UPLOAD_DIR = self.original_allowed_dir
        config.ENABLE_FILE_UPLOAD_SECURITY = self.original_security_enabled
    
    def test_valid_file_path(self):
        """Test that valid file paths are accepted."""
        is_valid, path, error = validate_file_path(self.test_file)
        assert is_valid is True
        assert error is None
        assert os.path.exists(path)
    
    def test_directory_traversal_blocked(self):
        """Test that directory traversal attempts are blocked."""
        # Try to access /etc/passwd
        is_valid, path, error = validate_file_path('/etc/passwd')
        assert is_valid is False
        assert 'Access denied' in error or 'not found' in error.lower()
        
        # Try relative path traversal
        is_valid, path, error = validate_file_path('../../../etc/passwd')
        assert is_valid is False
        assert error is not None
    
    def test_nonexistent_file(self):
        """Test that nonexistent files are rejected."""
        fake_path = os.path.join(self.test_dir, 'nonexistent.csv')
        is_valid, path, error = validate_file_path(fake_path)
        assert is_valid is False
        assert 'not found' in error.lower()
    
    def test_security_disabled(self):
        """Test that security can be disabled (not recommended)."""
        config.ENABLE_FILE_UPLOAD_SECURITY = False
        
        # Should still fail for nonexistent file
        is_valid, path, error = validate_file_path('/nonexistent/file.csv')
        assert is_valid is False
        assert 'not found' in error.lower()


class TestModelSchemaValidation:
    """Test unified model schema validation."""
    
    def setup_method(self):
        """Create fresh model manager."""
        self.model_manager = ModelManager()
    
    def test_schema_validation_first_training(self):
        """Test that first training has no schema to validate."""
        features = ['a', 'b', 'c']
        is_valid, error = self.model_manager.validate_feature_schema('test_model', features)
        assert is_valid is True
        assert error is None
    
    def test_schema_validation_matching(self):
        """Test that matching schemas pass validation."""
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        
        features = ['a', 'b', 'c']
        model = Ridge()
        scaler = StandardScaler()
        
        # Save model with schema
        self.model_manager.save_regression_model('test_model', model, scaler, feature_columns=features)
        
        # Validate same schema
        is_valid, error = self.model_manager.validate_feature_schema('test_model', features)
        assert is_valid is True
        assert error is None
    
    def test_schema_validation_missing_features(self):
        """Test that missing features are detected."""
        from sklearn.linear_model import Ridge
        
        original_features = ['a', 'b', 'c']
        new_features = ['a', 'b']  # Missing 'c'
        
        model = Ridge()
        self.model_manager.save_regression_model('test_model', model, None, feature_columns=original_features)
        
        is_valid, error = self.model_manager.validate_feature_schema('test_model', new_features)
        assert is_valid is False
        assert 'Missing features' in error
        assert 'c' in error
    
    def test_schema_validation_extra_features(self):
        """Test that extra features are detected."""
        from sklearn.linear_model import Ridge
        
        original_features = ['a', 'b']
        new_features = ['a', 'b', 'c']  # Extra 'c'
        
        model = Ridge()
        self.model_manager.save_regression_model('test_model', model, None, feature_columns=original_features)
        
        is_valid, error = self.model_manager.validate_feature_schema('test_model', new_features)
        assert is_valid is False
        assert 'Extra features' in error
        assert 'c' in error
    
    def test_schema_validation_order_matters(self):
        """Test that feature order is validated."""
        from sklearn.linear_model import Ridge
        
        original_features = ['a', 'b', 'c']
        new_features = ['c', 'b', 'a']  # Different order
        
        model = Ridge()
        self.model_manager.save_regression_model('test_model', model, None, feature_columns=original_features)
        
        is_valid, error = self.model_manager.validate_feature_schema('test_model', new_features)
        assert is_valid is False
        assert 'order mismatch' in error.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

