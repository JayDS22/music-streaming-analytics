"""
Cohort Analysis Module
======================

Analyze user cohorts based on signup date to understand:
- User retention over time
- Engagement patterns by cohort
- Churn prediction indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from loguru import logger

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class CohortAnalyzer:
    """
    Perform cohort analysis for user retention and engagement.
    
    Creates cohort matrices showing:
    - Retention rates by signup cohort
    - Engagement metrics over time
    - Churn indicators
    
    Example:
        >>> analyzer = CohortAnalyzer()
        >>> retention = analyzer.calculate_retention(users, sessions)
        >>> analyzer.plot_retention_heatmap(retention)
    """
    
    def __init__(self, period: str = 'monthly'):
        """
        Initialize cohort analyzer.
        
        Args:
            period: Cohort period ('weekly', 'monthly', 'quarterly')
        """
        self.period = period
        self.retention_matrix: Optional[pd.DataFrame] = None
        self.cohort_stats: Optional[pd.DataFrame] = None
    
    def calculate_retention(
        self,
        users: pd.DataFrame,
        sessions: pd.DataFrame,
        periods: int = 12
    ) -> pd.DataFrame:
        """
        Calculate retention matrix by signup cohort.
        
        Args:
            users: Users DataFrame with signup_date
            sessions: Sessions DataFrame with timestamp
            periods: Number of periods to analyze
            
        Returns:
            Retention matrix DataFrame
        """
        logger.info("Calculating retention matrix...")
        
        # Ensure datetime columns
        users = users.copy()
        sessions = sessions.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(users['signup_date']):
            users['signup_date'] = pd.to_datetime(users['signup_date'])
        if not pd.api.types.is_datetime64_any_dtype(sessions['timestamp']):
            sessions['timestamp'] = pd.to_datetime(sessions['timestamp'])
        
        # Create cohort labels
        if self.period == 'monthly':
            users['cohort'] = users['signup_date'].dt.to_period('M')
            sessions['activity_period'] = sessions['timestamp'].dt.to_period('M')
        elif self.period == 'weekly':
            users['cohort'] = users['signup_date'].dt.to_period('W')
            sessions['activity_period'] = sessions['timestamp'].dt.to_period('W')
        else:  # quarterly
            users['cohort'] = users['signup_date'].dt.to_period('Q')
            sessions['activity_period'] = sessions['timestamp'].dt.to_period('Q')
        
        # Merge to get cohort for each session
        sessions_with_cohort = sessions.merge(
            users[['user_id', 'cohort']],
            on='user_id',
            how='left'
        )
        
        # Get unique users per cohort per period
        cohort_activity = sessions_with_cohort.groupby(
            ['cohort', 'activity_period']
        )['user_id'].nunique().reset_index()
        cohort_activity.columns = ['cohort', 'activity_period', 'active_users']
        
        # Calculate period number (periods since signup)
        cohort_activity['period_number'] = (
            cohort_activity['activity_period'].astype(str).apply(lambda x: pd.Period(x)) -
            cohort_activity['cohort'].astype(str).apply(lambda x: pd.Period(x))
        ).apply(lambda x: x.n if hasattr(x, 'n') else 0)
        
        # Get cohort sizes
        cohort_sizes = users.groupby('cohort')['user_id'].nunique()
        
        # Create retention matrix
        retention_matrix = cohort_activity.pivot(
            index='cohort',
            columns='period_number',
            values='active_users'
        )
        
        # Limit to requested periods
        retention_matrix = retention_matrix.loc[
            :, retention_matrix.columns[:periods]
        ]
        
        # Convert to percentages
        for col in retention_matrix.columns:
            retention_matrix[col] = (
                retention_matrix[col] / cohort_sizes
            ) * 100
        
        self.retention_matrix = retention_matrix
        
        # Calculate cohort stats
        self.cohort_stats = pd.DataFrame({
            'cohort_size': cohort_sizes,
            'initial_retention': retention_matrix[0] if 0 in retention_matrix.columns else 0,
            'period_1_retention': retention_matrix[1] if 1 in retention_matrix.columns else 0,
            'period_3_retention': retention_matrix[3] if 3 in retention_matrix.columns else 0
        })
        
        logger.info(f"Calculated retention for {len(retention_matrix)} cohorts")
        
        return retention_matrix
    
    def calculate_cohort_engagement(
        self,
        users: pd.DataFrame,
        sessions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate engagement metrics by cohort.
        
        Args:
            users: Users DataFrame
            sessions: Sessions DataFrame
            
        Returns:
            Cohort engagement metrics
        """
        logger.info("Calculating cohort engagement metrics...")
        
        users = users.copy()
        sessions = sessions.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(users['signup_date']):
            users['signup_date'] = pd.to_datetime(users['signup_date'])
        
        # Create cohort labels
        if self.period == 'monthly':
            users['cohort'] = users['signup_date'].dt.to_period('M').astype(str)
        elif self.period == 'weekly':
            users['cohort'] = users['signup_date'].dt.to_period('W').astype(str)
        else:
            users['cohort'] = users['signup_date'].dt.to_period('Q').astype(str)
        
        # Merge sessions with user cohorts
        sessions_with_cohort = sessions.merge(
            users[['user_id', 'cohort']],
            on='user_id',
            how='left'
        )
        
        # Calculate metrics per cohort
        engagement = sessions_with_cohort.groupby('cohort').agg({
            'user_id': 'nunique',
            'session_id': 'count',
            'listen_duration_ms': ['sum', 'mean'],
            'skipped': 'mean'
        })
        
        engagement.columns = [
            'unique_users', 'total_sessions', 
            'total_listen_time_ms', 'avg_listen_duration_ms',
            'skip_rate'
        ]
        
        engagement['sessions_per_user'] = (
            engagement['total_sessions'] / engagement['unique_users']
        )
        engagement['listen_hours_per_user'] = (
            engagement['total_listen_time_ms'] / engagement['unique_users'] / (1000 * 60 * 60)
        )
        
        return engagement.reset_index()
    
    def identify_churn_risk(
        self,
        users: pd.DataFrame,
        sessions: pd.DataFrame,
        days_inactive: int = 30
    ) -> pd.DataFrame:
        """
        Identify users at risk of churn based on inactivity.
        
        Args:
            users: Users DataFrame
            sessions: Sessions DataFrame
            days_inactive: Days of inactivity to flag as at-risk
            
        Returns:
            DataFrame with churn risk indicators
        """
        logger.info(f"Identifying users inactive for {days_inactive}+ days...")
        
        sessions = sessions.copy()
        if not pd.api.types.is_datetime64_any_dtype(sessions['timestamp']):
            sessions['timestamp'] = pd.to_datetime(sessions['timestamp'])
        
        # Get last activity per user
        last_activity = sessions.groupby('user_id')['timestamp'].max().reset_index()
        last_activity.columns = ['user_id', 'last_activity']
        
        # Calculate days since last activity
        current_date = sessions['timestamp'].max()
        last_activity['days_inactive'] = (
            current_date - last_activity['last_activity']
        ).dt.days
        
        # Flag at-risk users
        last_activity['churn_risk'] = last_activity['days_inactive'] >= days_inactive
        
        # Merge with user info
        churn_analysis = users.merge(last_activity, on='user_id', how='left')
        
        # Users with no sessions are high risk
        churn_analysis['churn_risk'] = churn_analysis['churn_risk'].fillna(True)
        churn_analysis['days_inactive'] = churn_analysis['days_inactive'].fillna(999)
        
        logger.info(f"Identified {churn_analysis['churn_risk'].sum()} at-risk users")
        
        return churn_analysis
    
    def plot_retention_heatmap(
        self,
        retention_matrix: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (14, 8),
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot retention heatmap.
        
        Args:
            retention_matrix: Retention matrix (uses stored if None)
            figsize: Figure size
            save_path: Path to save figure
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Matplotlib/Seaborn not available for plotting")
            return
        
        if retention_matrix is None:
            retention_matrix = self.retention_matrix
        
        if retention_matrix is None:
            raise ValueError("No retention matrix available. Run calculate_retention first.")
        
        plt.figure(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            retention_matrix,
            annot=True,
            fmt='.1f',
            cmap='YlGnBu',
            vmin=0,
            vmax=100,
            cbar_kws={'label': 'Retention %'}
        )
        
        plt.title(f'{self.period.capitalize()} Cohort Retention Analysis', fontsize=14)
        plt.xlabel('Periods Since Signup')
        plt.ylabel('Signup Cohort')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved retention heatmap to {save_path}")
        
        plt.close()
    
    def get_retention_summary(self) -> Dict[str, float]:
        """
        Get summary statistics from retention analysis.
        
        Returns:
            Dictionary of summary metrics
        """
        if self.retention_matrix is None:
            raise ValueError("No retention matrix available. Run calculate_retention first.")
        
        return {
            'avg_initial_retention': self.retention_matrix[0].mean() if 0 in self.retention_matrix.columns else 0,
            'avg_period_1_retention': self.retention_matrix[1].mean() if 1 in self.retention_matrix.columns else 0,
            'avg_period_3_retention': self.retention_matrix[3].mean() if 3 in self.retention_matrix.columns else 0,
            'avg_period_6_retention': self.retention_matrix[6].mean() if 6 in self.retention_matrix.columns else 0,
            'best_cohort': str(self.retention_matrix[1].idxmax()) if 1 in self.retention_matrix.columns else None,
            'worst_cohort': str(self.retention_matrix[1].idxmin()) if 1 in self.retention_matrix.columns else None,
            'num_cohorts': len(self.retention_matrix)
        }


__all__ = ['CohortAnalyzer']
