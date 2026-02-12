"""
API Module
==========

External API integrations for the music streaming analytics platform.
"""

from .spotify_client import SpotifyClient, AudioFeatures

__all__ = ['SpotifyClient', 'AudioFeatures']
