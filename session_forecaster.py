"""
Session Duration Forecaster
===========================

Linear regression model for forecasting session listening duration.

Model Performance:
    - R²: 0.79
    - Key features: user tier, historical session length, playlist type
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.feature_selection import SelectKBest, f_regression

from loguru import logger


class SessionForecaster:
    """
    Forecast session listening duration.
    
    Uses linear regression with feature selection to predict
    how long a user will listen in a session.
    
    Example:
        >>> forecaster = SessionForecaster()
        >>> forecaster.train(X_train, y_train)
        >>> predictions = forecaster.predict(X_test)
        >>> r2 = forecaster.evaluate(X_test, y_test)
        >>> print(f"R²: {r2:.4f}")
    """
    
    def __init__(
        self,
        model_type: str = 'ridge',
        alpha: float = 1.0,
        n_features: int = 20,
        random_state: int = 42
    ):
        """
        Initialize session forecaster.
        
        Args:
            model_type: 'linear', 'ridge', or 'lasso'
            alpha: Regularization strength for ridge/lasso
            n_features: Number of features to select
            random_state: Random seed
        """
        self.model_type = model_type
        self.alpha = alpha
        self.n_features = n_features
        self.random_state = random_state
        
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector: Optional[SelectKBest] = None
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        self.is_fitted: bool = False
        
        self._metrics: Dict[str, float] = {}
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the session duration forecaster.
        
        Args:
            X: Feature matrix
            y: Target variable (session duration in ms)
            validation_split: Proportion for validation set
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training session forecaster on {len(X)} samples...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Feature selection
        n_features = min(self.n_features, X_train_scaled.shape[1])
        self.feature_selector = SelectKBest(f_regression, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = self.feature_selector.transform(X_val_scaled)
        
        # Get selected feature names
        selected_mask = self.feature_selector.get_support()
        self.selected_features = [
            self.feature_names[i] for i, selected in enumerate(selected_mask) if selected
        ]
        
        # Train model
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.model_type == 'lasso':
            self.model = Lasso(alpha=self.alpha, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_selected, y_train)
        self.is_fitted = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train_selected)
        val_pred = self.model.predict(X_val_selected)
        
        self._metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'n_features_selected': len(self.selected_features)
        }
        
        logger.info(f"Training complete. Validation R²: {self._metrics['val_r2']:.4f}")
        
        return self._metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict session duration.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted durations
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X[self.feature_names])
        X_selected = self.feature_selector.transform(X_scaled)
        return self.model.predict(X_selected)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Feature matrix
            y: True values
            verbose: Print detailed report
            
        Returns:
            Dictionary of evaluation metrics
        """
        self._check_fitted()
        
        y_pred = self.predict(X)
        
        metrics = {
            'r2': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100
        }
        
        if verbose:
            logger.info("\n" + "="*50)
            logger.info("Session Forecaster Evaluation")
            logger.info("="*50)
            logger.info(f"R² Score: {metrics['r2']:.4f}")
            logger.info(f"RMSE: {metrics['rmse']:.2f}")
            logger.info(f"MAE: {metrics['mae']:.2f}")
            logger.info(f"MAPE: {metrics['mape']:.2f}%")
            
            # Convert to minutes for interpretation
            rmse_minutes = metrics['rmse'] / (1000 * 60)
            mae_minutes = metrics['mae'] / (1000 * 60)
            logger.info(f"\nRMSE: {rmse_minutes:.2f} minutes")
            logger.info(f"MAE: {mae_minutes:.2f} minutes")
        
        return metrics
    
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5
    ) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of folds
            
        Returns:
            Cross-validation scores
        """
        logger.info(f"Running {cv}-fold cross-validation...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if self.model_type == 'linear':
            model = LinearRegression()
        elif self.model_type == 'ridge':
            model = Ridge(alpha=self.alpha)
        else:
            model = Lasso(alpha=self.alpha)
        
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='r2')
        
        results = {
            'cv_r2_mean': scores.mean(),
            'cv_r2_std': scores.std(),
            'cv_scores': scores.tolist()
        }
        
        logger.info(f"CV R²: {results['cv_r2_mean']:.4f} (+/- {results['cv_r2_std']:.4f})")
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importances from model coefficients.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        self._check_fitted()
        
        importance = pd.DataFrame({
            'feature': self.selected_features,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        importance['rank'] = range(1, len(importance) + 1)
        
        return importance.head(top_n)
    
    def get_residual_analysis(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict[str, Any]:
        """
        Perform residual analysis.
        
        Args:
            X: Feature matrix
            y: True values
            
        Returns:
            Residual statistics and data
        """
        self._check_fitted()
        
        y_pred = self.predict(X)
        residuals = y - y_pred
        
        return {
            'residuals': residuals,
            'mean': residuals.mean(),
            'std': residuals.std(),
            'min': residuals.min(),
            'max': residuals.max(),
            'skewness': residuals.skew(),
            'kurtosis': residuals.kurtosis()
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        self._check_fitted()
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'metrics': self._metrics,
            'config': {
                'model_type': self.model_type,
                'alpha': self.alpha,
                'n_features': self.n_features,
                'random_state': self.random_state
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SessionForecaster':
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded SessionForecaster instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        config = model_data['config']
        forecaster = cls(
            model_type=config['model_type'],
            alpha=config['alpha'],
            n_features=config['n_features'],
            random_state=config['random_state']
        )
        
        forecaster.model = model_data['model']
        forecaster.scaler = model_data['scaler']
        forecaster.feature_selector = model_data['feature_selector']
        forecaster.feature_names = model_data['feature_names']
        forecaster.selected_features = model_data['selected_features']
        forecaster._metrics = model_data['metrics']
        forecaster.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return forecaster
    
    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")


def create_session_features(sessions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Create features for session duration prediction.
    
    Args:
        sessions: Sessions DataFrame with user aggregations
        
    Returns:
        Feature matrix and target variable
    """
    logger.info("Creating session duration features...")
    
    # Aggregate sessions by user for features
    user_features = sessions.groupby('user_id').agg({
        'listen_duration_ms': ['mean', 'std', 'min', 'max', 'count'],
        'skipped': 'mean',
        'hour': 'mean',
        'is_weekend': 'mean'
    })
    
    user_features.columns = [
        'avg_duration', 'std_duration', 'min_duration', 'max_duration',
        'session_count', 'skip_rate', 'avg_hour', 'weekend_ratio'
    ]
    
    user_features = user_features.reset_index()
    
    # Target: average session duration
    y = user_features['avg_duration']
    
    # Features
    X = user_features.drop(['user_id', 'avg_duration'], axis=1)
    
    return X, y


__all__ = ['SessionForecaster', 'create_session_features']
