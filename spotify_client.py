"""
Spotify API Client
==================

Integration with Spotify Web API for extracting audio features.

Features extracted:
    - tempo: BPM of the track
    - energy: Intensity and activity (0.0 to 1.0)
    - danceability: Suitability for dancing (0.0 to 1.0)
    - valence: Musical positiveness (0.0 to 1.0)
    - acousticness, instrumentalness, liveness, speechiness
"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import time

from loguru import logger

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False
    logger.warning("spotipy not installed. Spotify API features will use mock data.")


@dataclass
class AudioFeatures:
    """Container for Spotify audio features."""
    track_id: str
    tempo: float
    energy: float
    danceability: float
    valence: float
    acousticness: float
    instrumentalness: float
    liveness: float
    speechiness: float
    loudness: float
    key: int
    mode: int
    time_signature: int
    duration_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'track_id': self.track_id,
            'tempo': self.tempo,
            'energy': self.energy,
            'danceability': self.danceability,
            'valence': self.valence,
            'acousticness': self.acousticness,
            'instrumentalness': self.instrumentalness,
            'liveness': self.liveness,
            'speechiness': self.speechiness,
            'loudness': self.loudness,
            'key': self.key,
            'mode': self.mode,
            'time_signature': self.time_signature,
            'duration_ms': self.duration_ms
        }


class SpotifyClient:
    """
    Client for interacting with Spotify Web API.
    
    Handles authentication and provides methods for:
    - Fetching audio features for tracks
    - Searching for tracks
    - Getting track metadata
    
    Example:
        >>> client = SpotifyClient()
        >>> features = client.get_audio_features("track_id_here")
        >>> print(f"Tempo: {features.tempo} BPM")
    """
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None
    ):
        """
        Initialize Spotify client.
        
        Args:
            client_id: Spotify API client ID (defaults to env var)
            client_secret: Spotify API client secret (defaults to env var)
        """
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        
        self._sp = None
        self._initialized = False
        
        if SPOTIPY_AVAILABLE and self.client_id and self.client_secret:
            self._initialize_client()
        else:
            logger.info("Running in mock mode - no Spotify credentials provided")
    
    def _initialize_client(self) -> None:
        """Initialize the Spotipy client with credentials."""
        try:
            auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self._sp = spotipy.Spotify(auth_manager=auth_manager)
            self._initialized = True
            logger.info("Spotify client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Spotify client: {e}")
            self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        """Check if client is properly initialized."""
        return self._initialized
    
    def get_audio_features(self, track_id: str) -> Optional[AudioFeatures]:
        """
        Get audio features for a single track.
        
        Args:
            track_id: Spotify track ID
            
        Returns:
            AudioFeatures object or None if failed
        """
        if not self._initialized:
            return self._mock_audio_features(track_id)
        
        try:
            features = self._sp.audio_features([track_id])[0]
            if features:
                return AudioFeatures(
                    track_id=track_id,
                    tempo=features['tempo'],
                    energy=features['energy'],
                    danceability=features['danceability'],
                    valence=features['valence'],
                    acousticness=features['acousticness'],
                    instrumentalness=features['instrumentalness'],
                    liveness=features['liveness'],
                    speechiness=features['speechiness'],
                    loudness=features['loudness'],
                    key=features['key'],
                    mode=features['mode'],
                    time_signature=features['time_signature'],
                    duration_ms=features['duration_ms']
                )
        except Exception as e:
            logger.error(f"Error fetching audio features for {track_id}: {e}")
        
        return None
    
    def get_audio_features_batch(
        self,
        track_ids: List[str],
        batch_size: int = 100
    ) -> List[AudioFeatures]:
        """
        Get audio features for multiple tracks.
        
        Args:
            track_ids: List of Spotify track IDs
            batch_size: Number of tracks per API call (max 100)
            
        Returns:
            List of AudioFeatures objects
        """
        results = []
        
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i + batch_size]
            
            if not self._initialized:
                results.extend([self._mock_audio_features(tid) for tid in batch])
            else:
                try:
                    features_list = self._sp.audio_features(batch)
                    for tid, features in zip(batch, features_list):
                        if features:
                            results.append(AudioFeatures(
                                track_id=tid,
                                tempo=features['tempo'],
                                energy=features['energy'],
                                danceability=features['danceability'],
                                valence=features['valence'],
                                acousticness=features['acousticness'],
                                instrumentalness=features['instrumentalness'],
                                liveness=features['liveness'],
                                speechiness=features['speechiness'],
                                loudness=features['loudness'],
                                key=features['key'],
                                mode=features['mode'],
                                time_signature=features['time_signature'],
                                duration_ms=features['duration_ms']
                            ))
                    # Rate limiting
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error fetching batch audio features: {e}")
            
            logger.info(f"Processed {min(i + batch_size, len(track_ids))}/{len(track_ids)} tracks")
        
        return results
    
    def search_tracks(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for tracks on Spotify.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of track dictionaries
        """
        if not self._initialized:
            return self._mock_search_results(query, limit)
        
        try:
            results = self._sp.search(q=query, type='track', limit=limit)
            return [
                {
                    'track_id': track['id'],
                    'name': track['name'],
                    'artist': track['artists'][0]['name'],
                    'album': track['album']['name'],
                    'duration_ms': track['duration_ms'],
                    'popularity': track['popularity']
                }
                for track in results['tracks']['items']
            ]
        except Exception as e:
            logger.error(f"Error searching tracks: {e}")
            return []
    
    def _mock_audio_features(self, track_id: str) -> AudioFeatures:
        """Generate mock audio features for testing."""
        import random
        random.seed(hash(track_id) % 2**32)
        
        return AudioFeatures(
            track_id=track_id,
            tempo=random.uniform(60, 180),
            energy=random.uniform(0, 1),
            danceability=random.uniform(0, 1),
            valence=random.uniform(0, 1),
            acousticness=random.uniform(0, 1),
            instrumentalness=random.uniform(0, 0.5),
            liveness=random.uniform(0, 0.3),
            speechiness=random.uniform(0, 0.3),
            loudness=random.uniform(-20, 0),
            key=random.randint(0, 11),
            mode=random.randint(0, 1),
            time_signature=4,
            duration_ms=random.randint(150000, 300000)
        )
    
    def _mock_search_results(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """Generate mock search results for testing."""
        import random
        results = []
        for i in range(limit):
            results.append({
                'track_id': f'mock_track_{hash(query)}_{i}',
                'name': f'Mock Track {i + 1}',
                'artist': f'Mock Artist {i % 5 + 1}',
                'album': f'Mock Album {i % 3 + 1}',
                'duration_ms': random.randint(150000, 300000),
                'popularity': random.randint(0, 100)
            })
        return results


# Module exports
__all__ = ['SpotifyClient', 'AudioFeatures']
