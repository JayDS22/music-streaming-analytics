"""
Dashboard Generator
===================

Generate visualizations and dashboard data for Tableau integration.
Tracks DAU/MAU, retention curves, skip rates.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class DashboardGenerator:
    """Generate dashboard visualizations and metrics."""
    
    def __init__(self, output_dir: str = "dashboards"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_dau_mau(
        self, sessions: pd.DataFrame, date_column: str = 'timestamp'
    ) -> pd.DataFrame:
        """Calculate Daily and Monthly Active Users."""
        sessions = sessions.copy()
        if not pd.api.types.is_datetime64_any_dtype(sessions[date_column]):
            sessions[date_column] = pd.to_datetime(sessions[date_column])
        
        sessions['date'] = sessions[date_column].dt.date
        sessions['month'] = sessions[date_column].dt.to_period('M')
        
        dau = sessions.groupby('date')['user_id'].nunique().reset_index()
        dau.columns = ['date', 'dau']
        
        mau = sessions.groupby('month')['user_id'].nunique().reset_index()
        mau.columns = ['month', 'mau']
        
        dau['month'] = pd.to_datetime(dau['date']).dt.to_period('M')
        metrics = dau.merge(mau, on='month', how='left')
        metrics['dau_mau_ratio'] = metrics['dau'] / metrics['mau']
        
        logger.info(f"Calculated DAU/MAU for {len(metrics)} days")
        return metrics
    
    def calculate_skip_rates(
        self, sessions: pd.DataFrame, tracks: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Calculate skip rates by various dimensions."""
        merged = sessions.merge(tracks[['track_id', 'genre']], on='track_id', how='left')
        merged['hour'] = pd.to_datetime(merged['timestamp']).dt.hour
        
        by_genre = merged.groupby('genre')['skipped'].mean().reset_index()
        by_genre.columns = ['genre', 'skip_rate']
        
        by_hour = merged.groupby('hour')['skipped'].mean().reset_index()
        by_hour.columns = ['hour', 'skip_rate']
        
        by_context = merged.groupby('context')['skipped'].mean().reset_index()
        by_context.columns = ['context', 'skip_rate']
        
        return {'by_genre': by_genre, 'by_hour': by_hour, 'by_context': by_context}
    
    def calculate_retention_curve(
        self, users: pd.DataFrame, sessions: pd.DataFrame, days: int = 30
    ) -> pd.DataFrame:
        """Calculate day-over-day retention curve."""
        users = users.copy()
        sessions = sessions.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(users['signup_date']):
            users['signup_date'] = pd.to_datetime(users['signup_date'])
        if not pd.api.types.is_datetime64_any_dtype(sessions['timestamp']):
            sessions['timestamp'] = pd.to_datetime(sessions['timestamp'])
        
        sessions['session_date'] = sessions['timestamp'].dt.date
        sessions_with_signup = sessions.merge(users[['user_id', 'signup_date']], on='user_id')
        sessions_with_signup['days_since_signup'] = (
            pd.to_datetime(sessions_with_signup['session_date']) - 
            sessions_with_signup['signup_date']
        ).dt.days
        
        total_users = users['user_id'].nunique()
        retention = []
        
        for day in range(days + 1):
            active = sessions_with_signup[
                sessions_with_signup['days_since_signup'] == day
            ]['user_id'].nunique()
            retention.append({'day': day, 'users': active, 'retention_rate': active / total_users})
        
        return pd.DataFrame(retention)
    
    def export_for_tableau(
        self, dau_mau: pd.DataFrame, skip_rates: Dict, retention: pd.DataFrame
    ) -> None:
        """Export data for Tableau dashboards."""
        dau_mau.to_csv(self.output_dir / 'dau_mau_metrics.csv', index=False)
        
        for name, df in skip_rates.items():
            df.to_csv(self.output_dir / f'skip_rates_{name}.csv', index=False)
        
        retention.to_csv(self.output_dir / 'retention_curve.csv', index=False)
        
        logger.info(f"Exported dashboard data to {self.output_dir}")
    
    def plot_all(
        self, dau_mau: pd.DataFrame, skip_rates: Dict, retention: pd.DataFrame
    ) -> None:
        """Generate all dashboard plots."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib not available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # DAU/MAU
        ax = axes[0, 0]
        ax.plot(range(len(dau_mau)), dau_mau['dau'], label='DAU')
        ax.set_title('Daily Active Users')
        ax.set_xlabel('Days')
        ax.set_ylabel('Users')
        
        # Skip rates by genre
        ax = axes[0, 1]
        skip_rates['by_genre'].plot(kind='bar', x='genre', y='skip_rate', ax=ax, legend=False)
        ax.set_title('Skip Rate by Genre')
        ax.set_ylabel('Skip Rate')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Skip rates by hour
        ax = axes[1, 0]
        ax.plot(skip_rates['by_hour']['hour'], skip_rates['by_hour']['skip_rate'])
        ax.set_title('Skip Rate by Hour')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Skip Rate')
        
        # Retention curve
        ax = axes[1, 1]
        ax.plot(retention['day'], retention['retention_rate'] * 100)
        ax.set_title('Retention Curve')
        ax.set_xlabel('Days Since Signup')
        ax.set_ylabel('Retention %')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'dashboard_preview.png', dpi=150)
        plt.close()
        
        logger.info(f"Saved dashboard preview to {self.output_dir / 'dashboard_preview.png'}")


__all__ = ['DashboardGenerator']
