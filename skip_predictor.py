"""
Skip Behavior Prediction Model
==============================

Logistic regression model for predicting whether a user will skip a track.

Model Performance:
    - AUC-ROC: 0.84
    - Key features: audio energy, user engagement history, time of day
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve
)

from loguru import logger


class SkipPredictor:
    """
    Predict whether a user will skip a track.
    
    Uses logistic regression with L2 regularization to predict
    skip probability based on audio features, user behavior,
    and contextual factors.
    
    Example:
        >>> predictor = SkipPredictor()
        >>> predictor.train(X_train, y_train)
        >>> predictions = predictor.predict_proba(X_test)
        >>> auc = predictor.evaluate(X_test, y_test)
        >>> print(f"AUC: {auc:.4f}")
    """
    
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced',
        random_state: int = 42
    ):
        """
        Initialize skip predictor.
        
        Args:
            C: Regularization strength (smaller = stronger)
            max_iter: Maximum iterations for solver
            class_weight: Handle class imbalance
            random_state: Random seed
        """
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        
        self._metrics: Dict[str, float] = {}
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validation_split: float = 0.2
    ) -> Dict[str, float]:
        """
        Train the skip prediction model.
        
        Args:
            X: Feature matrix
            y: Target variable (0/1 skip indicator)
            validation_split: Proportion for validation set
            
        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training skip predictor on {len(X)} samples...")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=validation_split,
            random_state=self.random_state,
            stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver='lbfgs',
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Calculate metrics
        train_pred = self.model.predict(X_train_scaled)
        train_prob = self.model.predict_proba(X_train_scaled)[:, 1]
        val_pred = self.model.predict(X_val_scaled)
        val_prob = self.model.predict_proba(X_val_scaled)[:, 1]
        
        self._metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_auc': roc_auc_score(y_train, train_prob),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_auc': roc_auc_score(y_val, val_prob),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred)
        }
        
        logger.info(f"Training complete. Validation AUC: {self._metrics['val_auc']:.4f}")
        
        return self._metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict skip labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Binary predictions (0/1)
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict skip probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability of skip for each sample
        """
        self._check_fitted()
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_scaled)[:, 1]
    
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
            y: True labels
            verbose: Print detailed report
            
        Returns:
            Dictionary of evaluation metrics
        """
        self._check_fitted()
        
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_prob),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
        
        if verbose:
            logger.info("\n" + "="*50)
            logger.info("Skip Prediction Model Evaluation")
            logger.info("="*50)
            logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1']:.4f}")
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(y, y_pred))
        
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
        
        model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        
        results = {
            'cv_auc_mean': scores.mean(),
            'cv_auc_std': scores.std(),
            'cv_scores': scores.tolist()
        }
        
        logger.info(f"CV AUC: {results['cv_auc_mean']:.4f} (+/- {results['cv_auc_std']:.4f})")
        
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
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)
        
        importance['rank'] = range(1, len(importance) + 1)
        
        return importance.head(top_n)
    
    def get_roc_curve(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, np.ndarray]:
        """
        Get ROC curve data.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary with fpr, tpr, thresholds
        """
        self._check_fitted()
        y_prob = self.predict_proba(X)
        fpr, tpr, thresholds = roc_curve(y, y_prob)
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc_score(y, y_prob)
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
            'feature_names': self.feature_names,
            'metrics': self._metrics,
            'config': {
                'C': self.C,
                'max_iter': self.max_iter,
                'class_weight': self.class_weight,
                'random_state': self.random_state
            }
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SkipPredictor':
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Loaded SkipPredictor instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        config = model_data['config']
        predictor = cls(
            C=config['C'],
            max_iter=config['max_iter'],
            class_weight=config['class_weight'],
            random_state=config['random_state']
        )
        
        predictor.model = model_data['model']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor._metrics = model_data['metrics']
        predictor.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")
        
        return predictor
    
    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")


__all__ = ['SkipPredictor']
