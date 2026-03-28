"""Predictive Models for Music Streaming Analytics"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import pickle
from pathlib import Path
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from loguru import logger


class SkipPredictor:
    """Logistic regression model for skip behavior prediction. Target AUC: 0.84"""
    
    def __init__(self, C: float = 1.0, max_iter: int = 1000, class_weight: str = 'balanced', random_state: int = 42):
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self.model: Optional[LogisticRegression] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self._metrics: Dict[str, float] = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, float]:
        logger.info(f"Training skip predictor on {len(X)} samples...")
        self.feature_names = list(X.columns)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=self.random_state, stratify=y)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter, class_weight=self.class_weight, random_state=self.random_state, solver='lbfgs')
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        train_prob = self.model.predict_proba(X_train_scaled)[:, 1]
        val_prob = self.model.predict_proba(X_val_scaled)[:, 1]
        val_pred = self.model.predict(X_val_scaled)
        
        self._metrics = {
            'train_auc': roc_auc_score(y_train, train_prob),
            'val_auc': roc_auc_score(y_val, val_prob),
            'val_accuracy': accuracy_score(y_val, val_pred),
            'val_precision': precision_score(y_val, val_pred),
            'val_recall': recall_score(y_val, val_pred),
            'val_f1': f1_score(y_val, val_pred)
        }
        logger.info(f"Training complete. Validation AUC: {self._metrics['val_auc']:.4f}")
        return self._metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_scaled = self.scaler.transform(X[self.feature_names])
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> Dict[str, float]:
        self._check_fitted()
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)
        metrics = {'accuracy': accuracy_score(y, y_pred), 'auc_roc': roc_auc_score(y, y_prob),
                   'precision': precision_score(y, y_pred), 'recall': recall_score(y, y_pred), 'f1': f1_score(y, y_pred)}
        if verbose:
            logger.info(f"Skip Predictor - AUC: {metrics['auc_roc']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        self._check_fitted()
        importance = pd.DataFrame({'feature': self.feature_names, 'coefficient': self.model.coef_[0], 'abs_coefficient': np.abs(self.model.coef_[0])})
        return importance.sort_values('abs_coefficient', ascending=False).head(top_n)
    
    def save(self, filepath: str):
        self._check_fitted()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'feature_names': self.feature_names, 'metrics': self._metrics}, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SkipPredictor':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        predictor = cls()
        predictor.model, predictor.scaler, predictor.feature_names, predictor._metrics = data['model'], data['scaler'], data['feature_names'], data['metrics']
        predictor.is_fitted = True
        return predictor
    
    def _check_fitted(self):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")


class SessionForecaster:
    """Linear regression model for session duration forecasting. Target R²: 0.79"""
    
    def __init__(self, model_type: str = 'ridge', alpha: float = 1.0, n_features: int = 20, random_state: int = 42):
        self.model_type = model_type
        self.alpha = alpha
        self.n_features = n_features
        self.random_state = random_state
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector = None
        self.feature_names: List[str] = []
        self.selected_features: List[str] = []
        self.is_fitted: bool = False
        self._metrics: Dict[str, float] = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series, validation_split: float = 0.2) -> Dict[str, float]:
        logger.info(f"Training session forecaster on {len(X)} samples...")
        self.feature_names = list(X.columns)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=self.random_state)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        n_features = min(self.n_features, X_train_scaled.shape[1])
        self.feature_selector = SelectKBest(f_regression, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = self.feature_selector.transform(X_val_scaled)
        
        self.selected_features = [self.feature_names[i] for i, sel in enumerate(self.feature_selector.get_support()) if sel]
        
        if self.model_type == 'ridge':
            self.model = Ridge(alpha=self.alpha, random_state=self.random_state)
        else:
            self.model = LinearRegression()
        
        self.model.fit(X_train_selected, y_train)
        self.is_fitted = True
        
        train_pred = self.model.predict(X_train_selected)
        val_pred = self.model.predict(X_val_selected)
        
        self._metrics = {
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'val_mae': mean_absolute_error(y_val, val_pred)
        }
        logger.info(f"Training complete. Validation R²: {self._metrics['val_r2']:.4f}")
        return self._metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        X_scaled = self.scaler.transform(X[self.feature_names])
        X_selected = self.feature_selector.transform(X_scaled)
        return self.model.predict(X_selected)
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> Dict[str, float]:
        self._check_fitted()
        y_pred = self.predict(X)
        metrics = {'r2': r2_score(y, y_pred), 'rmse': np.sqrt(mean_squared_error(y, y_pred)), 'mae': mean_absolute_error(y, y_pred)}
        if verbose:
            logger.info(f"Session Forecaster - R²: {metrics['r2']:.4f}, RMSE: {metrics['rmse']:.2f}")
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        self._check_fitted()
        importance = pd.DataFrame({'feature': self.selected_features, 'coefficient': self.model.coef_, 'abs_coefficient': np.abs(self.model.coef_)})
        return importance.sort_values('abs_coefficient', ascending=False).head(top_n)
    
    def save(self, filepath: str):
        self._check_fitted()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler, 'feature_selector': self.feature_selector, 
                        'feature_names': self.feature_names, 'selected_features': self.selected_features, 'metrics': self._metrics}, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'SessionForecaster':
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        forecaster = cls()
        forecaster.model, forecaster.scaler, forecaster.feature_selector = data['model'], data['scaler'], data['feature_selector']
        forecaster.feature_names, forecaster.selected_features, forecaster._metrics = data['feature_names'], data['selected_features'], data['metrics']
        forecaster.is_fitted = True
        return forecaster
    
    def _check_fitted(self):
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call train() first.")


__all__ = ['SkipPredictor', 'SessionForecaster']
