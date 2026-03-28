"""Synthetic Data Generator for Music Streaming Analytics"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class DataGeneratorConfig:
    num_users: int = 10000
    num_sessions: int = 1000000
    num_tracks: int = 50000
    num_playlists: int = 5000
    seed: int = 42
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"


class SyntheticDataGenerator:
    """Generate synthetic music streaming data."""
    
    GENRES = ['pop', 'rock', 'hip-hop', 'electronic', 'jazz', 'classical', 'r&b', 'country', 'indie', 'metal']
    TIERS = ['free', 'premium', 'family', 'student']
    COUNTRIES = ['US', 'UK', 'DE', 'FR', 'JP', 'BR', 'CA', 'AU', 'MX', 'ES']
    
    def __init__(self, config: Optional[DataGeneratorConfig] = None):
        self.config = config or DataGeneratorConfig()
        np.random.seed(self.config.seed)
        self.start_date = pd.to_datetime(self.config.start_date)
        self.end_date = pd.to_datetime(self.config.end_date)
        logger.info(f"Initialized generator with seed={self.config.seed}")
    
    def generate_users(self) -> pd.DataFrame:
        logger.info(f"Generating {self.config.num_users} users...")
        n = self.config.num_users
        date_range = (self.end_date - self.start_date).days
        
        users = pd.DataFrame({
            'user_id': [f"user_{i:07d}" for i in range(n)],
            'signup_date': [self.start_date + timedelta(days=int(np.random.randint(0, date_range))) for _ in range(n)],
            'tier': np.random.choice(self.TIERS, n, p=[0.5, 0.3, 0.15, 0.05]),
            'country': np.random.choice(self.COUNTRIES, n),
            'age_group': np.random.choice(['18-24', '25-34', '35-44', '45-54', '55+'], n, p=[0.25, 0.35, 0.20, 0.12, 0.08]),
            'preferred_genre': np.random.choice(self.GENRES, n),
            'skip_tendency': np.random.beta(2, 5, n)
        })
        logger.info(f"Generated {len(users)} users")
        return users
    
    def generate_tracks(self) -> pd.DataFrame:
        logger.info(f"Generating {self.config.num_tracks} tracks...")
        n = self.config.num_tracks
        
        tracks = pd.DataFrame({
            'track_id': [f"track_{i:07d}" for i in range(n)],
            'tempo': np.random.normal(120, 25, n).clip(60, 200),
            'energy': np.random.beta(2, 2, n),
            'danceability': np.random.beta(2.5, 2, n),
            'valence': np.random.beta(2, 2, n),
            'acousticness': np.random.beta(1.5, 3, n),
            'instrumentalness': np.random.beta(1, 5, n),
            'liveness': np.random.beta(1.5, 5, n),
            'speechiness': np.random.beta(1.5, 8, n),
            'loudness': np.random.normal(-8, 4, n).clip(-20, 0),
            'duration_ms': np.random.randint(120000, 480000, n),
            'genre': np.random.choice(self.GENRES, n),
            'artist_id': [f"artist_{i % 5000:05d}" for i in range(n)],
            'popularity': np.random.beta(1.5, 3, n) * 100
        })
        logger.info(f"Generated {len(tracks)} tracks")
        return tracks
    
    def generate_sessions(self, users: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Generating {self.config.num_sessions} sessions...")
        n = self.config.num_sessions
        user_ids = users['user_id'].values
        track_ids = tracks['track_id'].values
        user_skip = users.set_index('user_id')['skip_tendency'].to_dict()
        track_duration = tracks.set_index('track_id')['duration_ms'].to_dict()
        
        sessions = []
        for i in range(n):
            user_id = np.random.choice(user_ids)
            track_id = np.random.choice(track_ids)
            
            timestamp = self.start_date + timedelta(
                days=int(np.random.randint(0, (self.end_date - self.start_date).days)),
                hours=int(np.random.choice(range(24))),
                minutes=int(np.random.randint(0, 60))
            )
            
            skipped = np.random.random() < user_skip.get(user_id, 0.3)
            track_dur = track_duration.get(track_id, 200000)
            listen_dur = int(track_dur * np.random.beta(2, 5)) if skipped else int(track_dur * np.random.beta(8, 2))
            
            sessions.append({
                'session_id': f"sess_{i:010d}", 'user_id': user_id, 'track_id': track_id,
                'timestamp': timestamp, 'listen_duration_ms': listen_dur, 'track_duration_ms': track_dur,
                'skipped': skipped, 'context': np.random.choice(['playlist', 'album', 'radio', 'search', 'recommendation'], p=[0.4, 0.2, 0.15, 0.1, 0.15]),
                'device': np.random.choice(['mobile', 'desktop', 'tablet', 'smart_speaker'], p=[0.55, 0.30, 0.10, 0.05])
            })
        
        logger.info(f"Generated {len(sessions)} sessions")
        return pd.DataFrame(sessions)
    
    def generate_playlists(self, users: pd.DataFrame, tracks: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Generating {self.config.num_playlists} playlists...")
        playlists, playlist_tracks = [], []
        
        for i in range(self.config.num_playlists):
            playlist_id = f"playlist_{i:06d}"
            num_tracks = np.random.randint(10, 100)
            playlists.append({
                'playlist_id': playlist_id, 'user_id': np.random.choice(users['user_id']),
                'name': f"Playlist {i+1}", 'num_tracks': num_tracks,
                'created_date': self.start_date + timedelta(days=int(np.random.randint(0, (self.end_date - self.start_date).days)))
            })
            for pos, track_id in enumerate(np.random.choice(tracks['track_id'], num_tracks, replace=False)):
                playlist_tracks.append({'playlist_id': playlist_id, 'track_id': track_id, 'position': pos})
        
        logger.info(f"Generated {len(playlists)} playlists with {len(playlist_tracks)} track entries")
        return pd.DataFrame(playlists), pd.DataFrame(playlist_tracks)
    
    def generate_ab_test_data(self, users: pd.DataFrame, sessions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Generating A/B test data...")
        user_ids = users['user_id'].values
        assignments = pd.DataFrame({
            'user_id': user_ids, 'test_name': 'personalized_recommendations',
            'variant': np.random.choice(['control', 'treatment'], len(user_ids)),
            'assignment_date': self.start_date + timedelta(days=30)
        })
        
        user_sessions = sessions.groupby('user_id').agg({'session_id': 'count', 'skipped': 'mean'}).reset_index()
        user_sessions.columns = ['user_id', 'total_sessions', 'skip_rate']
        results = assignments.merge(user_sessions, on='user_id', how='left')
        results.loc[results['variant'] == 'treatment', 'skip_rate'] *= 0.95
        
        logger.info(f"Generated A/B test data for {len(results)} users")
        return assignments, results
    
    def generate_all(self) -> Dict[str, pd.DataFrame]:
        logger.info("Starting full data generation...")
        users = self.generate_users()
        tracks = self.generate_tracks()
        sessions = self.generate_sessions(users, tracks)
        playlists, playlist_tracks = self.generate_playlists(users, tracks)
        assignments, results = self.generate_ab_test_data(users, sessions)
        
        logger.info("Data generation complete!")
        return {
            'users': users, 'tracks': tracks, 'sessions': sessions,
            'playlists': playlists, 'playlist_tracks': playlist_tracks,
            'ab_test_assignments': assignments, 'ab_test_results': results
        }
    
    def save_data(self, data: Dict[str, pd.DataFrame], output_dir: str = "data/raw") -> None:
        import os
        os.makedirs(output_dir, exist_ok=True)
        for name, df in data.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} to {filepath}")


class DataLoader:
    """Load data from various sources."""
    
    def __init__(self, db_connection_string: Optional[str] = None):
        self.db_connection_string = db_connection_string
    
    def load_csv(self, filepath: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        logger.info(f"Loading data from {filepath}")
        return pd.read_csv(filepath, parse_dates=parse_dates)
    
    def load_all_csv(self, directory: str) -> Dict[str, pd.DataFrame]:
        from pathlib import Path
        data = {}
        for filepath in Path(directory).glob("*.csv"):
            data[filepath.stem] = self.load_csv(str(filepath))
        return data


class DataPreparer:
    """Prepare data for analysis."""
    
    @staticmethod
    def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
        if strategy == "drop":
            return df.dropna()
        elif strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df


__all__ = ['SyntheticDataGenerator', 'DataGeneratorConfig', 'DataLoader', 'DataPreparer']
