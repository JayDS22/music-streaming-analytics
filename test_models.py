"""
Tests for Prediction Models
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.skip_predictor import SkipPredictor
from src.models.session_forecaster import SessionForecaster


@pytest.fixture
def classification_data():
    """Generate sample classification data."""
    np.random.seed(42)
    n = 1000
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'feature_4': np.random.randn(n),
        'feature_5': np.random.randn(n)
    })
    
    # Create target with some correlation to features
    prob = 1 / (1 + np.exp(-(X['feature_1'] + 0.5 * X['feature_2'])))
    y = pd.Series((np.random.rand(n) < prob).astype(int))
    
    return X, y


@pytest.fixture
def regression_data():
    """Generate sample regression data."""
    np.random.seed(42)
    n = 1000
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'feature_4': np.random.randn(n),
        'feature_5': np.random.randn(n)
    })
    
    # Create target with linear relationship
    y = pd.Series(
        2 * X['feature_1'] + 1.5 * X['feature_2'] + np.random.randn(n) * 0.5
    )
    
    return X, y


class TestSkipPredictor:
    """Test SkipPredictor class."""
    
    def test_init(self):
        """Test initialization."""
        model = SkipPredictor()
        assert model.C == 1.0
        assert model.is_fitted == False
    
    def test_train(self, classification_data):
        """Test training."""
        X, y = classification_data
        model = SkipPredictor()
        
        metrics = model.train(X, y)
        
        assert model.is_fitted == True
        assert 'train_auc' in metrics
        assert 'val_auc' in metrics
        assert metrics['val_auc'] > 0.5  # Better than random
    
    def test_predict(self, classification_data):
        """Test prediction."""
        X, y = classification_data
        model = SkipPredictor()
        model.train(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_proba(self, classification_data):
        """Test probability prediction."""
        X, y = classification_data
        model = SkipPredictor()
        model.train(X, y)
        
        probas = model.predict_proba(X)
        
        assert len(probas) == len(X)
        assert all(0 <= p <= 1 for p in probas)
    
    def test_evaluate(self, classification_data):
        """Test evaluation."""
        X, y = classification_data
        model = SkipPredictor()
        model.train(X, y)
        
        metrics = model.evaluate(X, y, verbose=False)
        
        assert 'auc_roc' in metrics
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
    
    def test_feature_importance(self, classification_data):
        """Test feature importance."""
        X, y = classification_data
        model = SkipPredictor()
        model.train(X, y)
        
        importance = model.get_feature_importance(top_n=5)
        
        assert len(importance) == 5
        assert 'feature' in importance.columns
        assert 'coefficient' in importance.columns
    
    def test_save_load(self, classification_data):
        """Test save and load."""
        X, y = classification_data
        model = SkipPredictor()
        model.train(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model.pkl')
            model.save(filepath)
            
            loaded_model = SkipPredictor.load(filepath)
            
            assert loaded_model.is_fitted
            # Predictions should match
            np.testing.assert_array_equal(
                model.predict(X),
                loaded_model.predict(X)
            )
    
    def test_not_fitted_error(self, classification_data):
        """Test error when predicting before training."""
        X, _ = classification_data
        model = SkipPredictor()
        
        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)


class TestSessionForecaster:
    """Test SessionForecaster class."""
    
    def test_init(self):
        """Test initialization."""
        model = SessionForecaster()
        assert model.model_type == 'ridge'
        assert model.is_fitted == False
    
    def test_train(self, regression_data):
        """Test training."""
        X, y = regression_data
        model = SessionForecaster(n_features=5)
        
        metrics = model.train(X, y)
        
        assert model.is_fitted == True
        assert 'train_r2' in metrics
        assert 'val_r2' in metrics
        assert metrics['val_r2'] > 0  # Better than baseline
    
    def test_predict(self, regression_data):
        """Test prediction."""
        X, y = regression_data
        model = SessionForecaster(n_features=5)
        model.train(X, y)
        
        predictions = model.predict(X)
        
        assert len(predictions) == len(X)
    
    def test_evaluate(self, regression_data):
        """Test evaluation."""
        X, y = regression_data
        model = SessionForecaster(n_features=5)
        model.train(X, y)
        
        metrics = model.evaluate(X, y, verbose=False)
        
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
    
    def test_different_model_types(self, regression_data):
        """Test different model types."""
        X, y = regression_data
        
        for model_type in ['linear', 'ridge', 'lasso']:
            model = SessionForecaster(model_type=model_type, n_features=5)
            metrics = model.train(X, y)
            assert metrics['val_r2'] > 0
    
    def test_save_load(self, regression_data):
        """Test save and load."""
        X, y = regression_data
        model = SessionForecaster(n_features=5)
        model.train(X, y)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model.pkl')
            model.save(filepath)
            
            loaded_model = SessionForecaster.load(filepath)
            
            assert loaded_model.is_fitted
            np.testing.assert_array_almost_equal(
                model.predict(X),
                loaded_model.predict(X)
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
