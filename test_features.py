"""
Tests for Feature Engineering Module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import FeatureEngineer, create_skip_prediction_features


@pytest.fixture
def sample_users():
    """Create sample users DataFrame."""
    return pd.DataFrame({
        'user_id': [f'user_{i}' for i in range(100)],
        'signup_date': [datetime(2024, 1, 1) + timedelta(days=i % 30) for i in range(100)],
        'tier': np.random.choice(['free', 'premium'], 100),
        'country': np.random.choice(['US', 'UK', 'DE'], 100),
        'age_group': np.random.choice(['18-24', '25-34', '35-44'], 100)
    })


@pytest.fixture
def sample_tracks():
    """Create sample tracks DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        'track_id': [f'track_{i}' for i in range(500)],
        'tempo': np.random.normal(120, 20, 500),
        'energy': np.random.beta(2, 2, 500),
        'danceability': np.random.beta(2.5, 2, 500),
        'valence': np.random.beta(2, 2, 500),
        'acousticness': np.random.beta(1.5, 3, 500),
        'instrumentalness': np.random.beta(1, 5, 500),
        'liveness': np.random.beta(1.5, 5, 500),
        'speechiness': np.random.beta(1.5, 8, 500),
        'loudness': np.random.normal(-8, 4, 500),
        'genre': np.random.choice(['pop', 'rock', 'hip-hop', 'electronic', 'jazz'], 500),
        'duration_ms': np.random.randint(150000, 300000, 500)
    })


@pytest.fixture
def sample_sessions(sample_users, sample_tracks):
    """Create sample sessions DataFrame."""
    np.random.seed(42)
    n_sessions = 5000
    
    return pd.DataFrame({
        'session_id': [f'sess_{i}' for i in range(n_sessions)],
        'user_id': np.random.choice(sample_users['user_id'], n_sessions),
        'track_id': np.random.choice(sample_tracks['track_id'], n_sessions),
        'timestamp': [datetime(2024, 1, 1) + timedelta(hours=i % (24*60)) for i in range(n_sessions)],
        'listen_duration_ms': np.random.randint(30000, 240000, n_sessions),
        'track_duration_ms': np.random.randint(180000, 300000, n_sessions),
        'skipped': np.random.choice([True, False], n_sessions, p=[0.3, 0.7]),
        'context': np.random.choice(['playlist', 'album', 'radio', 'search'], n_sessions),
        'device': np.random.choice(['mobile', 'desktop', 'tablet'], n_sessions)
    })


class TestFeatureEngineer:
    """Test FeatureEngineer class."""
    
    def test_init(self):
        """Test initialization."""
        engineer = FeatureEngineer()
        assert engineer.feature_names == []
    
    def test_create_all_features(self, sample_sessions, sample_users, sample_tracks):
        """Test creating all features."""
        engineer = FeatureEngineer()
        features = engineer.create_all_features(sample_sessions, sample_users, sample_tracks)
        
        # Check output is DataFrame
        assert isinstance(features, pd.DataFrame)
        
        # Check we have features for each user
        assert len(features) == len(sample_users)
        
        # Check we created many features (should be 50+)
        assert len(features.columns) >= 30
        
        # Check key feature groups exist
        assert 'current_streak' in features.columns
        assert 'genre_entropy' in features.columns
        assert 'skip_rate' in features.columns
        assert 'engagement_score' in features.columns
    
    def test_feature_groups(self, sample_sessions, sample_users, sample_tracks):
        """Test feature importance groups."""
        engineer = FeatureEngineer()
        engineer.create_all_features(sample_sessions, sample_users, sample_tracks)
        
        groups = engineer.get_feature_importance_groups()
        
        assert 'streak_features' in groups
        assert 'genre_features' in groups
        assert 'temporal_features' in groups
        assert 'audio_features' in groups
    
    def test_no_nan_in_numeric_features(self, sample_sessions, sample_users, sample_tracks):
        """Test that numeric features have no NaN values."""
        engineer = FeatureEngineer()
        features = engineer.create_all_features(sample_sessions, sample_users, sample_tracks)
        
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        assert features[numeric_cols].isna().sum().sum() == 0


class TestSkipPredictionFeatures:
    """Test skip prediction feature creation."""
    
    def test_create_skip_features(self, sample_sessions, sample_tracks):
        """Test creating skip prediction features."""
        X, y, user_ids, session_ids = create_skip_prediction_features(
            sample_sessions, sample_tracks
        )
        
        # Check shapes
        assert len(X) == len(sample_sessions)
        assert len(y) == len(sample_sessions)
        
        # Check features exist
        assert 'tempo' in X.columns
        assert 'energy' in X.columns
        assert 'hour' in X.columns
        
        # Check target is binary
        assert set(y.unique()).issubset({0, 1})
    
    def test_no_nan_in_features(self, sample_sessions, sample_tracks):
        """Test no NaN values in features."""
        X, y, _, _ = create_skip_prediction_features(sample_sessions, sample_tracks)
        assert X.isna().sum().sum() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
