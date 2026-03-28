#!/usr/bin/env python3
"""
Music Streaming Analytics Platform - Main Entry Point

Usage:
    python main.py --full-pipeline
    python main.py --sessions 100000 --users 1000
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))

from src.data import SyntheticDataGenerator, DataGeneratorConfig
from src.features import FeatureEngineer, create_skip_prediction_features
from src.models import SkipPredictor, SessionForecaster
from src.analysis import CohortAnalyzer, FunnelAnalyzer
from src.ab_testing import ABTestFramework, run_ab_test_simulation
from src.visualization import DashboardGenerator
from src.utils import setup_logging, save_results


def generate_data(num_sessions: int = 100000, num_users: int = 1000) -> dict:
    """Generate synthetic data."""
    logger.info(f"Generating data: {num_users} users, {num_sessions} sessions")
    
    config = DataGeneratorConfig(
        num_users=num_users, num_sessions=num_sessions, num_tracks=5000, seed=42
    )
    generator = SyntheticDataGenerator(config)
    data = generator.generate_all()
    generator.save_data(data, "data/raw")
    return data


def run_feature_engineering(data: dict) -> dict:
    """Run feature engineering pipeline."""
    logger.info("Running feature engineering...")
    engineer = FeatureEngineer()
    user_features = engineer.create_all_features(data['sessions'], data['users'], data['tracks'])
    logger.info(f"Created {len(user_features.columns)} features")
    return {'user_features': user_features, 'feature_groups': engineer.get_feature_importance_groups()}


def train_models(data: dict) -> dict:
    """Train prediction models."""
    logger.info("Training models...")
    
    # Skip predictor
    X, y, _, _ = create_skip_prediction_features(data['sessions'], data['tracks'])
    skip_model = SkipPredictor(C=1.0, class_weight='balanced')
    skip_metrics = skip_model.train(X, y)
    logger.info(f"Skip Predictor AUC: {skip_metrics['val_auc']:.4f}")
    
    # Session forecaster
    sessions = data['sessions'].copy()
    sessions['hour'] = sessions['timestamp'].dt.hour
    sessions['is_weekend'] = (sessions['timestamp'].dt.dayofweek >= 5).astype(int)
    
    session_features = sessions.groupby('user_id').agg({
        'listen_duration_ms': ['count', 'std'], 'skipped': 'mean', 'hour': 'mean'
    })
    session_features.columns = ['session_count', 'duration_std', 'skip_rate', 'avg_hour']
    session_features = session_features.reset_index()
    
    y_duration = sessions.groupby('user_id')['listen_duration_ms'].mean()
    X_duration = session_features.drop('user_id', axis=1).fillna(0)
    y_duration = y_duration.loc[session_features['user_id']].values
    
    session_model = SessionForecaster(model_type='ridge', n_features=4)
    session_metrics = session_model.train(X_duration, y_duration)
    logger.info(f"Session Forecaster R²: {session_metrics['val_r2']:.4f}")
    
    return {'skip_metrics': skip_metrics, 'session_metrics': session_metrics}


def run_analysis(data: dict) -> dict:
    """Run cohort and funnel analysis."""
    logger.info("Running analysis...")
    
    cohort = CohortAnalyzer(period='monthly')
    retention = cohort.calculate_retention(data['users'], data['sessions'], periods=6)
    
    funnel = FunnelAnalyzer()
    funnel_metrics = funnel.analyze_playlist_completion(
        data['sessions'], data['playlists'], data['playlist_tracks']
    )
    logger.info(f"Playlist Drop-off (tracks 3-5): {funnel_metrics.get('drop_off_track_3_5', 0):.1%}")
    
    return {'cohort_stats': cohort.get_retention_summary(), 'funnel_metrics': funnel_metrics}


def run_full_pipeline(num_sessions: int = 100000, num_users: int = 1000):
    """Run complete analytics pipeline."""
    logger.info("=" * 60)
    logger.info("MUSIC STREAMING ANALYTICS PLATFORM")
    logger.info("=" * 60)
    
    data = generate_data(num_sessions, num_users)
    features = run_feature_engineering(data)
    models = train_models(data)
    analysis = run_analysis(data)
    ab_results = run_ab_test_simulation(n_users=10000, effect=0.05)
    
    # Generate dashboards
    dashboard = DashboardGenerator(output_dir="dashboards")
    dau_mau = dashboard.calculate_dau_mau(data['sessions'])
    skip_rates = dashboard.calculate_skip_rates(data['sessions'], data['tracks'])
    retention = dashboard.calculate_retention_curve(data['users'], data['sessions'])
    dashboard.export_for_tableau(dau_mau, skip_rates, retention)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Data: {num_users:,} users, {num_sessions:,} sessions")
    logger.info(f"Features: {len(features['user_features'].columns)} engineered")
    logger.info(f"Skip Predictor AUC: {models['skip_metrics']['val_auc']:.4f}")
    logger.info(f"Session Forecaster R²: {models['session_metrics']['val_r2']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Music Streaming Analytics Platform")
    parser.add_argument('--full-pipeline', action='store_true', help='Run complete pipeline')
    parser.add_argument('--sessions', type=int, default=100000, help='Number of sessions')
    parser.add_argument('--users', type=int, default=1000, help='Number of users')
    parser.add_argument('--debug', action='store_true', help='Debug logging')
    args = parser.parse_args()
    
    setup_logging(level="DEBUG" if args.debug else "INFO")
    run_full_pipeline(args.sessions, args.users)


if __name__ == "__main__":
    main()
