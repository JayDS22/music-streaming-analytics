"""Spotify API Client for Audio Feature Extraction"""

import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    SPOTIPY_AVAILABLE = False


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
    duration_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


class SpotifyClient:
    """Client for Spotify Web API - extracts audio features (tempo, energy, danceability, valence)."""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = client_id or os.getenv('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.getenv('SPOTIFY_CLIENT_SECRET')
        self._sp = None
        self._initialized = False
        
        if SPOTIPY_AVAILABLE and self.client_id and self.client_secret:
            try:
                auth_manager = SpotifyClientCredentials(client_id=self.client_id, client_secret=self.client_secret)
                self._sp = spotipy.Spotify(auth_manager=auth_manager)
                self._initialized = True
                logger.info("Spotify client initialized")
            except Exception as e:
                logger.warning(f"Spotify init failed: {e}. Using mock data.")
        else:
            logger.info("Running in mock mode - no Spotify credentials")
    
    def get_audio_features(self, track_id: str) -> Optional[AudioFeatures]:
        """Get audio features for a track."""
        if not self._initialized:
            return self._mock_audio_features(track_id)
        
        try:
            features = self._sp.audio_features([track_id])[0]
            if features:
                return AudioFeatures(
                    track_id=track_id, tempo=features['tempo'], energy=features['energy'],
                    danceability=features['danceability'], valence=features['valence'],
                    acousticness=features['acousticness'], instrumentalness=features['instrumentalness'],
                    liveness=features['liveness'], speechiness=features['speechiness'],
                    loudness=features['loudness'], duration_ms=features['duration_ms']
                )
        except Exception as e:
            logger.error(f"Error fetching features for {track_id}: {e}")
        return None
    
    def get_audio_features_batch(self, track_ids: List[str], batch_size: int = 100) -> List[AudioFeatures]:
        """Get audio features for multiple tracks."""
        results = []
        for i in range(0, len(track_ids), batch_size):
            batch = track_ids[i:i + batch_size]
            if not self._initialized:
                results.extend([self._mock_audio_features(tid) for tid in batch])
            else:
                try:
                    features_list = self._sp.audio_features(batch)
                    for tid, f in zip(batch, features_list):
                        if f:
                            results.append(AudioFeatures(
                                track_id=tid, tempo=f['tempo'], energy=f['energy'],
                                danceability=f['danceability'], valence=f['valence'],
                                acousticness=f['acousticness'], instrumentalness=f['instrumentalness'],
                                liveness=f['liveness'], speechiness=f['speechiness'],
                                loudness=f['loudness'], duration_ms=f['duration_ms']
                            ))
                except Exception as e:
                    logger.error(f"Batch error: {e}")
        return results
    
    def _mock_audio_features(self, track_id: str) -> AudioFeatures:
        """Generate mock audio features for testing."""
        import random
        random.seed(hash(track_id) % 2**32)
        return AudioFeatures(
            track_id=track_id, tempo=random.uniform(60, 180), energy=random.uniform(0, 1),
            danceability=random.uniform(0, 1), valence=random.uniform(0, 1),
            acousticness=random.uniform(0, 1), instrumentalness=random.uniform(0, 0.5),
            liveness=random.uniform(0, 0.3), speechiness=random.uniform(0, 0.3),
            loudness=random.uniform(-20, 0), duration_ms=random.randint(150000, 300000)
        )


__all__ = ['SpotifyClient', 'AudioFeatures']
