"""
Synthetic Data Generator
========================

Generate realistic synthetic data for testing and development.
Simulates 1M+ listening sessions with realistic patterns.

Data Generated:
    - Users: Demographics, subscription tiers, signup dates
    - Tracks: Audio features, genres, artists
    - Sessions: Listening events, skips, durations
    - Playlists: Track collections, completion rates
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import hashlib

from loguru import logger


@dataclass
class DataGeneratorConfig:
    """Configuration for data generation."""
    num_users: int = 10000
    num_sessions: int = 1000000
    num_tracks: int = 50000
    num_playlists: int = 5000
    seed: int = 42
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"


class SyntheticDataGenerator:
    """
    Generate synthetic music streaming data.
    
    Creates realistic datasets including:
    - User profiles with demographics
    - Track catalog with audio features
    - Listening sessions with skip behavior
    - Playlist data with engagement metrics
    
    Example:
        >>> generator = SyntheticDataGenerator(seed=42)
        >>> users, tracks, sessions, playlists = generator.generate_all()
        >>> print(f"Generated {len(sessions)} sessions")
    """
    
    GENRES = [
        'pop', 'rock', 'hip-hop', 'electronic', 'jazz',
        'classical', 'r&b', 'country', 'indie', 'metal'
    ]
    
    TIERS = ['free', 'premium', 'family', 'student']
    TIER_WEIGHTS = [0.5, 0.3, 0.15, 0.05]
    
    COUNTRIES = ['US', 'UK', 'DE', 'FR', 'JP', 'BR', 'CA', 'AU', 'MX', 'ES']
    
    def __init__(self, config: Optional[DataGeneratorConfig] = None):
        """
        Initialize the data generator.
        
        Args:
            config: Configuration object (uses defaults if not provided)
        """
        self.config = config or DataGeneratorConfig()
        np.random.seed(self.config.seed)
        
        self.start_date = pd.to_datetime(self.config.start_date)
        self.end_date = pd.to_datetime(self.config.end_date)
        
        logger.info(f"Initialized generator with seed={self.config.seed}")
    
    def generate_users(self) -> pd.DataFrame:
        """
        Generate synthetic user data.
        
        Returns:
            DataFrame with user profiles
        """
        logger.info(f"Generating {self.config.num_users} users...")
        
        # Generate user IDs
        user_ids = [f"user_{i:07d}" for i in range(self.config.num_users)]
        
        # Signup dates (weighted towards recent)
        date_range = (self.end_date - self.start_date).days
        signup_weights = np.exp(np.linspace(-2, 0, date_range))
        signup_weights /= signup_weights.sum()
        
        signup_days = np.random.choice(date_range, self.config.num_users, p=signup_weights)
        signup_dates = [self.start_date + timedelta(days=int(d)) for d in signup_days]
        
        # User attributes
        users = pd.DataFrame({
            'user_id': user_ids,
            'signup_date': signup_dates,
            'tier': np.random.choice(self.TIERS, self.config.num_users, p=self.TIER_WEIGHTS),
            'country': np.random.choice(self.COUNTRIES, self.config.num_users),
            'age_group': np.random.choice(
                ['18-24', '25-34', '35-44', '45-54', '55+'],
                self.config.num_users,
                p=[0.25, 0.35, 0.20, 0.12, 0.08]
            ),
            'gender': np.random.choice(
                ['M', 'F', 'Other'],
                self.config.num_users,
                p=[0.48, 0.48, 0.04]
            )
        })
        
        # Add user preferences (affects behavior)
        users['preferred_genre'] = np.random.choice(self.GENRES, self.config.num_users)
        users['avg_session_length_pref'] = np.random.lognormal(3.5, 0.5, self.config.num_users)
        users['skip_tendency'] = np.random.beta(2, 5, self.config.num_users)
        
        logger.info(f"Generated {len(users)} users")
        return users
    
    def generate_tracks(self) -> pd.DataFrame:
        """
        Generate synthetic track catalog with audio features.
        
        Returns:
            DataFrame with track data including audio features
        """
        logger.info(f"Generating {self.config.num_tracks} tracks...")
        
        track_ids = [f"track_{i:07d}" for i in range(self.config.num_tracks)]
        
        # Generate audio features (similar to Spotify)
        tracks = pd.DataFrame({
            'track_id': track_ids,
            'tempo': np.random.normal(120, 25, self.config.num_tracks).clip(60, 200),
            'energy': np.random.beta(2, 2, self.config.num_tracks),
            'danceability': np.random.beta(2.5, 2, self.config.num_tracks),
            'valence': np.random.beta(2, 2, self.config.num_tracks),
            'acousticness': np.random.beta(1.5, 3, self.config.num_tracks),
            'instrumentalness': np.random.beta(1, 5, self.config.num_tracks),
            'liveness': np.random.beta(1.5, 5, self.config.num_tracks),
            'speechiness': np.random.beta(1.5, 8, self.config.num_tracks),
            'loudness': np.random.normal(-8, 4, self.config.num_tracks).clip(-20, 0),
            'duration_ms': np.random.lognormal(12.2, 0.3, self.config.num_tracks).astype(int),
            'genre': np.random.choice(self.GENRES, self.config.num_tracks),
            'artist_id': [f"artist_{i % 5000:05d}" for i in range(self.config.num_tracks)],
            'release_year': np.random.choice(
                range(1990, 2025),
                self.config.num_tracks,
                p=np.exp(np.linspace(-2, 0, 35)) / np.exp(np.linspace(-2, 0, 35)).sum()
            ),
            'popularity': np.random.beta(1.5, 3, self.config.num_tracks) * 100
        })
        
        # Ensure duration is reasonable (2-8 minutes)
        tracks['duration_ms'] = tracks['duration_ms'].clip(120000, 480000)
        
        logger.info(f"Generated {len(tracks)} tracks")
        return tracks
    
    def generate_sessions(
        self,
        users: pd.DataFrame,
        tracks: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate synthetic listening sessions.
        
        Args:
            users: User DataFrame
            tracks: Track DataFrame
            
        Returns:
            DataFrame with session data
        """
        logger.info(f"Generating {self.config.num_sessions} sessions...")
        
        sessions = []
        user_ids = users['user_id'].values
        track_ids = tracks['track_id'].values
        
        # Create lookup for user preferences
        user_prefs = users.set_index('user_id')[['skip_tendency', 'preferred_genre', 'avg_session_length_pref']].to_dict('index')
        track_genres = tracks.set_index('track_id')['genre'].to_dict()
        track_energy = tracks.set_index('track_id')['energy'].to_dict()
        
        for i in range(self.config.num_sessions):
            # Select user (weighted by activity level)
            user_id = np.random.choice(user_ids)
            user_pref = user_prefs[user_id]
            
            # Select track (weighted by popularity and genre match)
            track_id = np.random.choice(track_ids)
            
            # Generate session timestamp
            timestamp = self.start_date + timedelta(
                days=int(np.random.randint(0, (self.end_date - self.start_date).days)),
                hours=int(np.random.choice(range(24), p=self._get_hour_distribution())),
                minutes=int(np.random.randint(0, 60))
            )
            
            # Calculate skip probability based on user tendency and track match
            genre_match = 1.0 if track_genres[track_id] == user_pref['preferred_genre'] else 0.7
            base_skip_prob = user_pref['skip_tendency']
            skip_prob = base_skip_prob / genre_match
            
            # Energy affects skip rate (very high/low energy more likely skipped)
            energy = track_energy[track_id]
            if energy > 0.8 or energy < 0.2:
                skip_prob *= 1.2
            
            skipped = np.random.random() < skip_prob
            
            # Listen duration (shorter if skipped)
            track_duration = tracks[tracks['track_id'] == track_id]['duration_ms'].values[0]
            if skipped:
                listen_duration_ms = int(track_duration * np.random.beta(2, 5))
            else:
                listen_duration_ms = int(track_duration * np.random.beta(8, 2))
            
            sessions.append({
                'session_id': f"sess_{i:010d}",
                'user_id': user_id,
                'track_id': track_id,
                'timestamp': timestamp,
                'listen_duration_ms': listen_duration_ms,
                'track_duration_ms': track_duration,
                'skipped': skipped,
                'skip_time_ms': listen_duration_ms if skipped else None,
                'context': np.random.choice(
                    ['playlist', 'album', 'radio', 'search', 'recommendation'],
                    p=[0.4, 0.2, 0.15, 0.1, 0.15]
                ),
                'device': np.random.choice(
                    ['mobile', 'desktop', 'tablet', 'smart_speaker'],
                    p=[0.55, 0.30, 0.10, 0.05]
                )
            })
            
            if (i + 1) % 100000 == 0:
                logger.info(f"Generated {i + 1}/{self.config.num_sessions} sessions")
        
        sessions_df = pd.DataFrame(sessions)
        logger.info(f"Generated {len(sessions_df)} sessions")
        return sessions_df
    
    def generate_playlists(
        self,
        users: pd.DataFrame,
        tracks: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate synthetic playlists and playlist tracks.
        
        Args:
            users: User DataFrame
            tracks: Track DataFrame
            
        Returns:
            Tuple of (playlists DataFrame, playlist_tracks DataFrame)
        """
        logger.info(f"Generating {self.config.num_playlists} playlists...")
        
        playlists = []
        playlist_tracks = []
        
        user_ids = users['user_id'].values
        track_ids = tracks['track_id'].values
        
        for i in range(self.config.num_playlists):
            playlist_id = f"playlist_{i:06d}"
            user_id = np.random.choice(user_ids)
            num_tracks = np.random.randint(10, 100)
            
            playlists.append({
                'playlist_id': playlist_id,
                'user_id': user_id,
                'name': f"Playlist {i + 1}",
                'created_date': self.start_date + timedelta(
                    days=np.random.randint(0, (self.end_date - self.start_date).days)
                ),
                'num_tracks': num_tracks,
                'is_public': np.random.random() > 0.3
            })
            
            # Add tracks to playlist
            selected_tracks = np.random.choice(track_ids, num_tracks, replace=False)
            for pos, track_id in enumerate(selected_tracks):
                playlist_tracks.append({
                    'playlist_id': playlist_id,
                    'track_id': track_id,
                    'position': pos
                })
        
        playlists_df = pd.DataFrame(playlists)
        playlist_tracks_df = pd.DataFrame(playlist_tracks)
        
        logger.info(f"Generated {len(playlists_df)} playlists with {len(playlist_tracks_df)} track entries")
        return playlists_df, playlist_tracks_df
    
    def generate_ab_test_data(
        self,
        users: pd.DataFrame,
        sessions: pd.DataFrame,
        test_name: str = "personalized_recommendations"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate A/B test assignment and results data.
        
        Args:
            users: User DataFrame
            sessions: Sessions DataFrame
            test_name: Name of the test
            
        Returns:
            Tuple of (test_assignment DataFrame, test_results DataFrame)
        """
        logger.info("Generating A/B test data...")
        
        # Randomly assign users to control/treatment
        user_ids = users['user_id'].values
        assignments = np.random.choice(['control', 'treatment'], len(user_ids))
        
        test_assignments = pd.DataFrame({
            'user_id': user_ids,
            'test_name': test_name,
            'variant': assignments,
            'assignment_date': self.start_date + timedelta(days=30)
        })
        
        # Calculate metrics for each user
        user_sessions = sessions.groupby('user_id').agg({
            'session_id': 'count',
            'skipped': 'mean',
            'listen_duration_ms': 'sum'
        }).reset_index()
        user_sessions.columns = ['user_id', 'total_sessions', 'skip_rate', 'total_listen_time']
        
        # Merge with assignments
        test_results = test_assignments.merge(user_sessions, on='user_id', how='left')
        
        # Simulate treatment effect (5% improvement in listen-through rate)
        treatment_mask = test_results['variant'] == 'treatment'
        test_results.loc[treatment_mask, 'skip_rate'] *= 0.95
        
        logger.info(f"Generated A/B test data for {len(test_results)} users")
        return test_assignments, test_results
    
    def generate_all(self) -> Dict[str, pd.DataFrame]:
        """
        Generate all synthetic datasets.
        
        Returns:
            Dictionary containing all generated DataFrames
        """
        logger.info("Starting full data generation...")
        
        users = self.generate_users()
        tracks = self.generate_tracks()
        sessions = self.generate_sessions(users, tracks)
        playlists, playlist_tracks = self.generate_playlists(users, tracks)
        test_assignments, test_results = self.generate_ab_test_data(users, sessions)
        
        data = {
            'users': users,
            'tracks': tracks,
            'sessions': sessions,
            'playlists': playlists,
            'playlist_tracks': playlist_tracks,
            'ab_test_assignments': test_assignments,
            'ab_test_results': test_results
        }
        
        logger.info("Data generation complete!")
        return data
    
    def save_data(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: str = "data/raw"
    ) -> None:
        """
        Save generated data to CSV files.
        
        Args:
            data: Dictionary of DataFrames
            output_dir: Output directory path
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, df in data.items():
            filepath = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {name} to {filepath}")
    
    def _get_hour_distribution(self) -> np.ndarray:
        """Get probability distribution for hours of day."""
        # Peak listening in evening (18-22), low overnight
        hours = np.array([
            0.01, 0.005, 0.003, 0.002, 0.003, 0.01,  # 0-5
            0.02, 0.04, 0.06, 0.06, 0.05, 0.05,      # 6-11
            0.06, 0.06, 0.05, 0.06, 0.07, 0.08,      # 12-17
            0.10, 0.11, 0.10, 0.08, 0.05, 0.02       # 18-23
        ])
        return hours / hours.sum()


def main():
    """CLI entry point for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic music streaming data")
    parser.add_argument("--sessions", type=int, default=1000000, help="Number of sessions")
    parser.add_argument("--users", type=int, default=10000, help="Number of users")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    
    args = parser.parse_args()
    
    config = DataGeneratorConfig(
        num_users=args.users,
        num_sessions=args.sessions,
        seed=args.seed
    )
    
    generator = SyntheticDataGenerator(config)
    data = generator.generate_all()
    generator.save_data(data, args.output)


if __name__ == "__main__":
    main()
