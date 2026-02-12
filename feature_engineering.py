"""
Feature Engineering Module
==========================

Comprehensive feature engineering for music streaming analytics.
Creates 50+ user engagement features across multiple categories:

Categories:
    - Listening Streaks: Active days, streak patterns
    - Genre Diversity: Entropy, variety, exploration
    - Playlist Behavior: Completion, engagement, skips
    - Session Metrics: Duration, frequency, timing
    - Temporal Patterns: Time of day, weekday, seasonality
    - Audio Preferences: Tempo, energy, valence preferences
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats
from loguru import logger


class FeatureEngineer:
    """
    Create comprehensive user engagement features.
    
    Generates 50+ features across 6 categories for user behavior modeling.
    
    Example:
        >>> engineer = FeatureEngineer()
        >>> features = engineer.create_all_features(sessions_df, users_df, tracks_df)
        >>> print(f"Created {len(features.columns)} features")
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.feature_names: List[str] = []
        
    def create_all_features(
        self,
        sessions: pd.DataFrame,
        users: pd.DataFrame,
        tracks: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create all features for users.
        
        Args:
            sessions: Sessions DataFrame
            users: Users DataFrame
            tracks: Tracks DataFrame
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(sessions['timestamp']):
            sessions['timestamp'] = pd.to_datetime(sessions['timestamp'])
        
        # Merge track audio features into sessions
        sessions_with_tracks = sessions.merge(
            tracks[['track_id', 'tempo', 'energy', 'danceability', 'valence', 
                    'acousticness', 'genre', 'duration_ms']],
            on='track_id',
            how='left',
            suffixes=('', '_track')
        )
        
        # Create feature groups
        streak_features = self._create_listening_streak_features(sessions)
        genre_features = self._create_genre_diversity_features(sessions_with_tracks)
        playlist_features = self._create_playlist_behavior_features(sessions)
        session_features = self._create_session_features(sessions_with_tracks)
        temporal_features = self._create_temporal_features(sessions)
        audio_features = self._create_audio_preference_features(sessions_with_tracks)
        engagement_features = self._create_engagement_features(sessions)
        
        # Combine all features
        all_features = streak_features.merge(genre_features, on='user_id', how='outer')
        all_features = all_features.merge(playlist_features, on='user_id', how='outer')
        all_features = all_features.merge(session_features, on='user_id', how='outer')
        all_features = all_features.merge(temporal_features, on='user_id', how='outer')
        all_features = all_features.merge(audio_features, on='user_id', how='outer')
        all_features = all_features.merge(engagement_features, on='user_id', how='outer')
        
        # Add user demographic features
        all_features = all_features.merge(
            users[['user_id', 'tier', 'country', 'age_group']],
            on='user_id',
            how='left'
        )
        
        # Encode categorical variables
        all_features = self._encode_categorical_features(all_features)
        
        # Fill missing values
        numeric_cols = all_features.select_dtypes(include=[np.number]).columns
        all_features[numeric_cols] = all_features[numeric_cols].fillna(0)
        
        self.feature_names = all_features.columns.tolist()
        logger.info(f"Created {len(all_features.columns)} features for {len(all_features)} users")
        
        return all_features
    
    def _create_listening_streak_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Create listening streak features."""
        logger.info("Creating listening streak features...")
        
        sessions = sessions.copy()
        sessions['date'] = sessions['timestamp'].dt.date
        user_dates = sessions.groupby('user_id')['date'].apply(set).reset_index()
        
        features = []
        for _, row in user_dates.iterrows():
            user_id = row['user_id']
            dates = sorted(row['date'])
            
            if len(dates) == 0:
                features.append({
                    'user_id': user_id,
                    'current_streak': 0, 'max_streak': 0, 'avg_streak': 0,
                    'streak_count': 0, 'active_days': 0, 'active_days_ratio': 0
                })
                continue
            
            streaks = []
            current_streak = 1
            
            for i in range(1, len(dates)):
                if (dates[i] - dates[i-1]).days == 1:
                    current_streak += 1
                else:
                    streaks.append(current_streak)
                    current_streak = 1
            streaks.append(current_streak)
            
            date_range = (max(dates) - min(dates)).days + 1
            
            features.append({
                'user_id': user_id,
                'current_streak': streaks[-1] if streaks else 0,
                'max_streak': max(streaks) if streaks else 0,
                'avg_streak': np.mean(streaks) if streaks else 0,
                'streak_count': len(streaks),
                'active_days': len(dates),
                'active_days_ratio': len(dates) / date_range if date_range > 0 else 0
            })
        
        return pd.DataFrame(features)
    
    def _create_genre_diversity_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Create genre diversity features."""
        logger.info("Creating genre diversity features...")
        
        user_genre_counts = sessions.groupby(['user_id', 'genre']).size().unstack(fill_value=0)
        
        features = []
        for user_id in user_genre_counts.index:
            counts = user_genre_counts.loc[user_id].values
            total = counts.sum()
            
            if total == 0:
                features.append({
                    'user_id': user_id, 'genre_entropy': 0, 'genre_count': 0,
                    'top_genre_ratio': 0, 'genre_exploration_rate': 0, 'genre_concentration': 1
                })
                continue
            
            probs = counts / total
            probs = probs[probs > 0]
            
            entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 1 else 0
            genre_count = np.sum(counts > 0)
            top_genre_ratio = counts.max() / total
            hhi = np.sum(probs ** 2)
            exploration_rate = np.mean(probs < 0.05) if len(probs) > 1 else 0
            
            features.append({
                'user_id': user_id, 'genre_entropy': entropy, 'genre_count': genre_count,
                'top_genre_ratio': top_genre_ratio, 'genre_exploration_rate': exploration_rate,
                'genre_concentration': hhi
            })
        
        return pd.DataFrame(features)
    
    def _create_playlist_behavior_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Create playlist behavior features."""
        logger.info("Creating playlist behavior features...")
        
        playlist_sessions = sessions[sessions['context'] == 'playlist']
        total_sessions = sessions.groupby('user_id').size()
        playlist_counts = playlist_sessions.groupby('user_id').size()
        playlist_skips = playlist_sessions.groupby('user_id')['skipped'].mean()
        
        features = []
        for user_id in total_sessions.index:
            total = total_sessions.get(user_id, 0)
            playlist_count = playlist_counts.get(user_id, 0)
            skip_rate = playlist_skips.get(user_id, 0)
            
            features.append({
                'user_id': user_id,
                'playlist_session_ratio': playlist_count / total if total > 0 else 0,
                'playlist_track_count': playlist_count,
                'playlist_skip_rate': skip_rate,
                'playlist_completion_tendency': 1 - skip_rate
            })
        
        return pd.DataFrame(features)
    
    def _create_session_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Create session-level features."""
        logger.info("Creating session features...")
        
        features = sessions.groupby('user_id').agg({
            'listen_duration_ms': ['sum', 'mean', 'std'],
            'track_duration_ms': 'mean',
            'skipped': ['mean', 'sum'],
            'session_id': 'nunique',
            'track_id': 'nunique'
        }).reset_index()
        
        features.columns = [
            'user_id', 'total_listen_time_ms', 'avg_listen_duration_ms', 'std_listen_duration_ms',
            'avg_track_duration_ms', 'skip_rate', 'total_skips', 'session_count', 'unique_tracks'
        ]
        
        features['avg_listen_ratio'] = (
            features['avg_listen_duration_ms'] / features['avg_track_duration_ms']
        ).clip(0, 1)
        
        total_tracks = sessions.groupby('user_id').size()
        features['avg_tracks_per_session'] = features['user_id'].map(
            total_tracks / features.set_index('user_id')['session_count']
        )
        
        features['total_listen_hours'] = features['total_listen_time_ms'] / (1000 * 60 * 60)
        
        return features
    
    def _create_temporal_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Create temporal pattern features."""
        logger.info("Creating temporal features...")
        
        sessions = sessions.copy()
        sessions['hour'] = sessions['timestamp'].dt.hour
        sessions['day_of_week'] = sessions['timestamp'].dt.dayofweek
        sessions['is_weekend'] = sessions['day_of_week'] >= 5
        
        sessions['time_bucket'] = pd.cut(
            sessions['hour'], bins=[-1, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        )
        
        user_counts = sessions.groupby('user_id').size()
        
        time_buckets = sessions.groupby(['user_id', 'time_bucket']).size().unstack(fill_value=0)
        for bucket in ['night', 'morning', 'afternoon', 'evening']:
            if bucket not in time_buckets.columns:
                time_buckets[bucket] = 0
        
        time_buckets = time_buckets.div(user_counts, axis=0).fillna(0)
        time_buckets.columns = [f'{col}_ratio' for col in time_buckets.columns]
        
        weekend_counts = sessions.groupby('user_id')['is_weekend'].sum()
        weekend_ratio = (weekend_counts / user_counts).fillna(0)
        
        peak_hours = sessions.groupby('user_id')['hour'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12
        )
        
        hour_counts = sessions.groupby(['user_id', 'hour']).size().unstack(fill_value=0)
        hour_probs = hour_counts.div(hour_counts.sum(axis=1), axis=0)
        hour_entropy = -((hour_probs * np.log2(hour_probs.replace(0, 1))).sum(axis=1))
        
        features = time_buckets.reset_index()
        features['weekend_ratio'] = features['user_id'].map(weekend_ratio)
        features['peak_hour'] = features['user_id'].map(peak_hours)
        features['hour_entropy'] = features['user_id'].map(hour_entropy)
        
        return features
    
    def _create_audio_preference_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Create audio preference features."""
        logger.info("Creating audio preference features...")
        
        audio_cols = ['tempo', 'energy', 'danceability', 'valence', 'acousticness']
        
        features = sessions.groupby('user_id')[audio_cols].agg(['mean', 'std']).reset_index()
        
        new_cols = ['user_id']
        for col in audio_cols:
            new_cols.extend([f'avg_{col}_pref', f'{col}_variance'])
        features.columns = new_cols
        
        variance_cols = [col for col in features.columns if 'variance' in col]
        features[variance_cols] = features[variance_cols].fillna(0)
        
        features['high_energy_ratio'] = sessions.groupby('user_id').apply(
            lambda x: (x['energy'] > 0.7).mean()
        ).values
        
        features['mellow_music_ratio'] = sessions.groupby('user_id').apply(
            lambda x: ((x['energy'] < 0.4) & (x['acousticness'] > 0.5)).mean()
        ).values
        
        return features
    
    def _create_engagement_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Create engagement features."""
        logger.info("Creating engagement features...")
        
        user_dates = sessions.groupby('user_id')['timestamp'].agg(['min', 'max', 'count'])
        user_dates.columns = ['first_listen', 'last_listen', 'total_events']
        
        current_date = sessions['timestamp'].max()
        
        user_dates['days_since_first_listen'] = (current_date - user_dates['first_listen']).dt.days
        user_dates['days_since_last_listen'] = (current_date - user_dates['last_listen']).dt.days
        
        user_dates['listening_span_days'] = (
            user_dates['last_listen'] - user_dates['first_listen']
        ).dt.days + 1
        user_dates['avg_daily_sessions'] = (
            user_dates['total_events'] / user_dates['listening_span_days']
        ).clip(0, 100)
        
        recency_score = 1 - (user_dates['days_since_last_listen'] / 365).clip(0, 1)
        frequency_score = (user_dates['avg_daily_sessions'] / 10).clip(0, 1)
        volume_score = (np.log1p(user_dates['total_events']) / 10).clip(0, 1)
        
        user_dates['engagement_score'] = (
            recency_score * 0.3 + frequency_score * 0.4 + volume_score * 0.3
        )
        
        features = user_dates.reset_index()[
            ['user_id', 'days_since_first_listen', 'days_since_last_listen',
             'avg_daily_sessions', 'engagement_score', 'total_events']
        ]
        
        return features
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features as numeric."""
        df = df.copy()
        
        tier_map = {'free': 0, 'student': 1, 'premium': 2, 'family': 3}
        if 'tier' in df.columns:
            df['tier_encoded'] = df['tier'].map(tier_map).fillna(0)
        
        age_map = {'18-24': 0, '25-34': 1, '35-44': 2, '45-54': 3, '55+': 4}
        if 'age_group' in df.columns:
            df['age_group_encoded'] = df['age_group'].map(age_map).fillna(1)
        
        if 'country' in df.columns:
            country_freq = df['country'].value_counts(normalize=True).to_dict()
            df['country_frequency'] = df['country'].map(country_freq).fillna(0.1)
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get feature names grouped by category."""
        return {
            'streak_features': [
                'current_streak', 'max_streak', 'avg_streak', 
                'streak_count', 'active_days', 'active_days_ratio'
            ],
            'genre_features': [
                'genre_entropy', 'genre_count', 'top_genre_ratio',
                'genre_exploration_rate', 'genre_concentration'
            ],
            'playlist_features': [
                'playlist_session_ratio', 'playlist_track_count',
                'playlist_skip_rate', 'playlist_completion_tendency'
            ],
            'session_features': [
                'total_listen_time_ms', 'avg_listen_duration_ms', 
                'std_listen_duration_ms', 'skip_rate', 'total_skips',
                'session_count', 'unique_tracks', 'avg_listen_ratio',
                'total_listen_hours'
            ],
            'temporal_features': [
                'night_ratio', 'morning_ratio', 'afternoon_ratio',
                'evening_ratio', 'weekend_ratio', 'peak_hour', 'hour_entropy'
            ],
            'audio_features': [
                'avg_tempo_pref', 'tempo_variance', 'avg_energy_pref',
                'energy_variance', 'avg_danceability_pref', 'danceability_variance',
                'avg_valence_pref', 'valence_variance', 'avg_acousticness_pref',
                'acousticness_variance', 'high_energy_ratio', 'mellow_music_ratio'
            ],
            'engagement_features': [
                'days_since_first_listen', 'days_since_last_listen',
                'avg_daily_sessions', 'engagement_score', 'total_events'
            ],
            'demographic_features': [
                'tier_encoded', 'age_group_encoded', 'country_frequency'
            ]
        }


def create_skip_prediction_features(
    sessions: pd.DataFrame,
    tracks: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Create features specifically for skip prediction.
    
    Returns feature matrix with one row per session.
    """
    logger.info("Creating skip prediction features...")
    
    df = sessions.merge(tracks, on='track_id', how='left')
    
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
    
    context_dummies = pd.get_dummies(df['context'], prefix='context')
    df = pd.concat([df, context_dummies], axis=1)
    
    device_dummies = pd.get_dummies(df['device'], prefix='device')
    df = pd.concat([df, device_dummies], axis=1)
    
    user_skip_rate = df.groupby('user_id')['skipped'].transform('mean')
    user_listen_count = df.groupby('user_id')['session_id'].transform('count')
    
    df['user_historical_skip_rate'] = user_skip_rate
    df['user_listen_count'] = user_listen_count
    
    track_skip_rate = df.groupby('track_id')['skipped'].transform('mean')
    df['track_skip_rate'] = track_skip_rate
    
    feature_cols = [
        'tempo', 'energy', 'danceability', 'valence', 'acousticness',
        'instrumentalness', 'liveness', 'speechiness', 'loudness',
        'hour', 'is_weekend', 'is_morning', 'is_evening',
        'user_historical_skip_rate', 'user_listen_count', 'track_skip_rate'
    ]
    
    feature_cols.extend([c for c in df.columns if c.startswith('context_')])
    feature_cols.extend([c for c in df.columns if c.startswith('device_')])
    
    X = df[feature_cols].fillna(0)
    y = df['skipped'].astype(int)
    
    logger.info(f"Created {len(feature_cols)} features for {len(X)} sessions")
    
    return X, y, df['user_id'], df['session_id']


__all__ = ['FeatureEngineer', 'create_skip_prediction_features']
