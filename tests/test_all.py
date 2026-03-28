"""Tests for Music Streaming Analytics Platform"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============== FIXTURES ==============
@pytest.fixture
def sample_users():
    return pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(100)],
        'signup_date': [datetime(2024, 1, 1) + timedelta(days=i % 30) for i in range(100)],
        'tier': np.random.choice(['free', 'premium'], 100),
        'country': np.random.choice(['US', 'UK'], 100),
        'age_group': np.random.choice(['18-24', '25-34'], 100)
    })

@pytest.fixture
def sample_tracks():
    np.random.seed(42)
    return pd.DataFrame({
        'track_id': [f'track_{i}' for i in range(500)],
        'tempo': np.random.normal(120, 20, 500),
        'energy': np.random.beta(2, 2, 500),
        'danceability': np.random.beta(2, 2, 500),
        'valence': np.random.beta(2, 2, 500),
        'acousticness': np.random.beta(2, 2, 500),
        'instrumentalness': np.random.beta(1, 5, 500),
        'liveness': np.random.beta(2, 5, 500),
        'speechiness': np.random.beta(2, 8, 500),
        'loudness': np.random.normal(-8, 4, 500),
        'genre': np.random.choice(['pop', 'rock', 'hip-hop'], 500),
        'duration_ms': np.random.randint(150000, 300000, 500)
    })

@pytest.fixture
def sample_sessions(sample_users, sample_tracks):
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        'session_id': [f'sess_{i}' for i in range(n)],
        'user_id': np.random.choice(sample_users['user_id'], n),
        'track_id': np.random.choice(sample_tracks['track_id'], n),
        'timestamp': [datetime(2024, 1, 1) + timedelta(hours=i % 720) for i in range(n)],
        'listen_duration_ms': np.random.randint(30000, 240000, n),
        'track_duration_ms': np.random.randint(180000, 300000, n),
        'skipped': np.random.choice([True, False], n, p=[0.3, 0.7]),
        'context': np.random.choice(['playlist', 'album', 'radio'], n),
        'device': np.random.choice(['mobile', 'desktop'], n)
    })


# ============== FEATURE TESTS ==============
class TestFeatureEngineer:
    def test_create_all_features(self, sample_sessions, sample_users, sample_tracks):
        from src.features import FeatureEngineer
        engineer = FeatureEngineer()
        features = engineer.create_all_features(sample_sessions, sample_users, sample_tracks)
        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_users)
        assert len(features.columns) >= 30

    def test_skip_prediction_features(self, sample_sessions, sample_tracks):
        from src.features import create_skip_prediction_features
        X, y, _, _ = create_skip_prediction_features(sample_sessions, sample_tracks)
        assert len(X) == len(sample_sessions)
        assert X.isna().sum().sum() == 0


# ============== MODEL TESTS ==============
class TestSkipPredictor:
    def test_train_and_predict(self):
        from src.models import SkipPredictor
        np.random.seed(42)
        X = pd.DataFrame({'f1': np.random.randn(1000), 'f2': np.random.randn(1000)})
        y = pd.Series((np.random.rand(1000) > 0.5).astype(int))
        
        model = SkipPredictor()
        metrics = model.train(X, y)
        assert 'val_auc' in metrics
        assert model.is_fitted
        
        preds = model.predict(X)
        assert len(preds) == len(X)

class TestSessionForecaster:
    def test_train_and_predict(self):
        from src.models import SessionForecaster
        np.random.seed(42)
        X = pd.DataFrame({'f1': np.random.randn(500), 'f2': np.random.randn(500)})
        y = pd.Series(X['f1'] * 2 + np.random.randn(500) * 0.5)
        
        model = SessionForecaster(n_features=2)
        metrics = model.train(X, y)
        assert 'val_r2' in metrics
        assert model.is_fitted


# ============== AB TEST TESTS ==============
class TestABFramework:
    def test_create_and_analyze(self):
        from src.ab_testing import ABTestFramework
        np.random.seed(42)
        
        ab = ABTestFramework()
        ab.create_experiment("test", [f"u{i}" for i in range(500)], [f"u{i}" for i in range(500, 1000)])
        
        data = pd.DataFrame({
            'user_id': [f"u{i}" for i in range(1000)],
            'metric': np.concatenate([np.random.normal(0.3, 0.1, 500), np.random.normal(0.35, 0.1, 500)])
        })
        
        results = ab.analyze_results("test", data, "metric")
        assert results.control_n == 500
        assert results.treatment_n == 500
        assert 0 <= results.p_value <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
