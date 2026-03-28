"""Feature Engineering Module - 50+ User Engagement Features"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger


class FeatureEngineer:
    """Create 50+ user engagement features across 8 categories."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.feature_names: List[str] = []
    
    def create_all_features(self, sessions: pd.DataFrame, users: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting feature engineering...")
        
        if not pd.api.types.is_datetime64_any_dtype(sessions['timestamp']):
            sessions['timestamp'] = pd.to_datetime(sessions['timestamp'])
        
        sessions_with_tracks = sessions.merge(
            tracks[['track_id', 'tempo', 'energy', 'danceability', 'valence', 'acousticness', 'genre', 'duration_ms']],
            on='track_id', how='left', suffixes=('', '_track')
        )
        
        streak_features = self._create_streak_features(sessions)
        genre_features = self._create_genre_features(sessions_with_tracks)
        playlist_features = self._create_playlist_features(sessions)
        session_features = self._create_session_features(sessions_with_tracks)
        temporal_features = self._create_temporal_features(sessions)
        audio_features = self._create_audio_features(sessions_with_tracks)
        engagement_features = self._create_engagement_features(sessions)
        
        all_features = streak_features
        for df in [genre_features, playlist_features, session_features, temporal_features, audio_features, engagement_features]:
            all_features = all_features.merge(df, on='user_id', how='outer')
        
        all_features = all_features.merge(users[['user_id', 'tier', 'country', 'age_group']], on='user_id', how='left')
        all_features = self._encode_categorical(all_features)
        
        numeric_cols = all_features.select_dtypes(include=[np.number]).columns
        all_features[numeric_cols] = all_features[numeric_cols].fillna(0)
        
        self.feature_names = all_features.columns.tolist()
        logger.info(f"Created {len(all_features.columns)} features for {len(all_features)} users")
        return all_features
    
    def _create_streak_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating listening streak features...")
        sessions = sessions.copy()
        sessions['date'] = sessions['timestamp'].dt.date
        user_dates = sessions.groupby('user_id')['date'].apply(set).reset_index()
        
        features = []
        for _, row in user_dates.iterrows():
            dates = sorted(row['date'])
            if len(dates) == 0:
                features.append({'user_id': row['user_id'], 'current_streak': 0, 'max_streak': 0, 'avg_streak': 0, 'active_days': 0, 'active_days_ratio': 0})
                continue
            
            streaks, current = [], 1
            for i in range(1, len(dates)):
                if (dates[i] - dates[i-1]).days == 1:
                    current += 1
                else:
                    streaks.append(current)
                    current = 1
            streaks.append(current)
            
            date_range = (max(dates) - min(dates)).days + 1
            features.append({
                'user_id': row['user_id'], 'current_streak': streaks[-1], 'max_streak': max(streaks),
                'avg_streak': np.mean(streaks), 'active_days': len(dates),
                'active_days_ratio': len(dates) / date_range if date_range > 0 else 0
            })
        return pd.DataFrame(features)
    
    def _create_genre_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating genre diversity features...")
        user_genre_counts = sessions.groupby(['user_id', 'genre']).size().unstack(fill_value=0)
        
        features = []
        for user_id in user_genre_counts.index:
            counts = user_genre_counts.loc[user_id].values
            total = counts.sum()
            if total == 0:
                features.append({'user_id': user_id, 'genre_entropy': 0, 'genre_count': 0, 'top_genre_ratio': 0, 'genre_concentration': 1})
                continue
            
            probs = counts[counts > 0] / total
            entropy = -np.sum(probs * np.log2(probs)) if len(probs) > 1 else 0
            features.append({
                'user_id': user_id, 'genre_entropy': entropy, 'genre_count': np.sum(counts > 0),
                'top_genre_ratio': counts.max() / total, 'genre_concentration': np.sum(probs ** 2)
            })
        return pd.DataFrame(features)
    
    def _create_playlist_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
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
                'user_id': user_id, 'playlist_session_ratio': playlist_count / total if total > 0 else 0,
                'playlist_track_count': playlist_count, 'playlist_skip_rate': skip_rate
            })
        return pd.DataFrame(features)
    
    def _create_session_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating session features...")
        features = sessions.groupby('user_id').agg({
            'listen_duration_ms': ['sum', 'mean', 'std'],
            'track_duration_ms': 'mean', 'skipped': ['mean', 'sum'],
            'session_id': 'nunique', 'track_id': 'nunique'
        }).reset_index()
        
        features.columns = ['user_id', 'total_listen_time_ms', 'avg_listen_duration_ms', 'std_listen_duration_ms',
                           'avg_track_duration_ms', 'skip_rate', 'total_skips', 'session_count', 'unique_tracks']
        features['avg_listen_ratio'] = (features['avg_listen_duration_ms'] / features['avg_track_duration_ms']).clip(0, 1)
        features['total_listen_hours'] = features['total_listen_time_ms'] / (1000 * 60 * 60)
        return features
    
    def _create_temporal_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating temporal features...")
        sessions = sessions.copy()
        sessions['hour'] = sessions['timestamp'].dt.hour
        sessions['is_weekend'] = sessions['timestamp'].dt.dayofweek >= 5
        sessions['time_bucket'] = pd.cut(sessions['hour'], bins=[-1, 6, 12, 18, 24], labels=['night', 'morning', 'afternoon', 'evening'])
        
        user_counts = sessions.groupby('user_id').size()
        time_buckets = sessions.groupby(['user_id', 'time_bucket']).size().unstack(fill_value=0)
        for bucket in ['night', 'morning', 'afternoon', 'evening']:
            if bucket not in time_buckets.columns:
                time_buckets[bucket] = 0
        time_buckets = time_buckets.div(user_counts, axis=0).fillna(0)
        time_buckets.columns = [f'{col}_ratio' for col in time_buckets.columns]
        
        weekend_ratio = (sessions.groupby('user_id')['is_weekend'].sum() / user_counts).fillna(0)
        peak_hours = sessions.groupby('user_id')['hour'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 12)
        
        features = time_buckets.reset_index()
        features['weekend_ratio'] = features['user_id'].map(weekend_ratio)
        features['peak_hour'] = features['user_id'].map(peak_hours)
        return features
    
    def _create_audio_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating audio preference features...")
        audio_cols = ['tempo', 'energy', 'danceability', 'valence', 'acousticness']
        features = sessions.groupby('user_id')[audio_cols].agg(['mean', 'std']).reset_index()
        
        new_cols = ['user_id']
        for col in audio_cols:
            new_cols.extend([f'avg_{col}_pref', f'{col}_variance'])
        features.columns = new_cols
        
        variance_cols = [col for col in features.columns if 'variance' in col]
        features[variance_cols] = features[variance_cols].fillna(0)
        
        features['high_energy_ratio'] = sessions.groupby('user_id').apply(lambda x: (x['energy'] > 0.7).mean()).values
        return features
    
    def _create_engagement_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        logger.info("Creating engagement features...")
        user_dates = sessions.groupby('user_id')['timestamp'].agg(['min', 'max', 'count'])
        user_dates.columns = ['first_listen', 'last_listen', 'total_events']
        
        current_date = sessions['timestamp'].max()
        user_dates['days_since_first_listen'] = (current_date - user_dates['first_listen']).dt.days
        user_dates['days_since_last_listen'] = (current_date - user_dates['last_listen']).dt.days
        user_dates['listening_span_days'] = (user_dates['last_listen'] - user_dates['first_listen']).dt.days + 1
        user_dates['avg_daily_sessions'] = (user_dates['total_events'] / user_dates['listening_span_days']).clip(0, 100)
        
        recency = 1 - (user_dates['days_since_last_listen'] / 365).clip(0, 1)
        frequency = (user_dates['avg_daily_sessions'] / 10).clip(0, 1)
        volume = (np.log1p(user_dates['total_events']) / 10).clip(0, 1)
        user_dates['engagement_score'] = recency * 0.3 + frequency * 0.4 + volume * 0.3
        
        return user_dates.reset_index()[['user_id', 'days_since_first_listen', 'days_since_last_listen', 'avg_daily_sessions', 'engagement_score', 'total_events']]
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'tier' in df.columns:
            df['tier_encoded'] = df['tier'].map({'free': 0, 'student': 1, 'premium': 2, 'family': 3}).fillna(0)
        if 'age_group' in df.columns:
            df['age_group_encoded'] = df['age_group'].map({'18-24': 0, '25-34': 1, '35-44': 2, '45-54': 3, '55+': 4}).fillna(1)
        if 'country' in df.columns:
            country_freq = df['country'].value_counts(normalize=True).to_dict()
            df['country_frequency'] = df['country'].map(country_freq).fillna(0.1)
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        return {
            'streak_features': ['current_streak', 'max_streak', 'avg_streak', 'active_days', 'active_days_ratio'],
            'genre_features': ['genre_entropy', 'genre_count', 'top_genre_ratio', 'genre_concentration'],
            'playlist_features': ['playlist_session_ratio', 'playlist_track_count', 'playlist_skip_rate'],
            'session_features': ['total_listen_time_ms', 'avg_listen_duration_ms', 'skip_rate', 'session_count', 'unique_tracks'],
            'temporal_features': ['night_ratio', 'morning_ratio', 'afternoon_ratio', 'evening_ratio', 'weekend_ratio', 'peak_hour'],
            'audio_features': ['avg_tempo_pref', 'avg_energy_pref', 'avg_danceability_pref', 'avg_valence_pref', 'high_energy_ratio'],
            'engagement_features': ['days_since_first_listen', 'days_since_last_listen', 'avg_daily_sessions', 'engagement_score'],
            'demographic_features': ['tier_encoded', 'age_group_encoded', 'country_frequency']
        }


def create_skip_prediction_features(sessions: pd.DataFrame, tracks: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Create features for skip prediction model."""
    logger.info("Creating skip prediction features...")
    
    df = sessions.merge(tracks, on='track_id', how='left')
    df['hour'] = df['timestamp'].dt.hour
    df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
    
    context_dummies = pd.get_dummies(df['context'], prefix='context')
    device_dummies = pd.get_dummies(df['device'], prefix='device')
    df = pd.concat([df, context_dummies, device_dummies], axis=1)
    
    df['user_historical_skip_rate'] = df.groupby('user_id')['skipped'].transform('mean')
    df['user_listen_count'] = df.groupby('user_id')['session_id'].transform('count')
    df['track_skip_rate'] = df.groupby('track_id')['skipped'].transform('mean')
    
    feature_cols = ['tempo', 'energy', 'danceability', 'valence', 'acousticness', 'instrumentalness',
                    'liveness', 'speechiness', 'loudness', 'hour', 'is_weekend', 'is_morning', 'is_evening',
                    'user_historical_skip_rate', 'user_listen_count', 'track_skip_rate']
    feature_cols.extend([c for c in df.columns if c.startswith('context_') or c.startswith('device_')])
    
    X = df[feature_cols].fillna(0)
    y = df['skipped'].astype(int)
    
    logger.info(f"Created {len(feature_cols)} features for {len(X)} sessions")
    return X, y, df['user_id'], df['session_id']


__all__ = ['FeatureEngineer', 'create_skip_prediction_features']
