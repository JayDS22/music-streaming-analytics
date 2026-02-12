"""
Data Loader Module
==================

Utilities for loading data from various sources:
- CSV files
- PostgreSQL database
- Parquet files

Handles data validation and preprocessing.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

try:
    from sqlalchemy import create_engine
    from sqlalchemy.engine import Engine
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


class DataLoader:
    """
    Load data from various sources.
    
    Supports:
    - CSV files
    - PostgreSQL database
    - Parquet files
    
    Example:
        >>> loader = DataLoader()
        >>> sessions = loader.load_csv("data/raw/sessions.csv")
        >>> all_data = loader.load_all_csv("data/raw")
    """
    
    def __init__(self, db_connection_string: Optional[str] = None):
        """
        Initialize data loader.
        
        Args:
            db_connection_string: PostgreSQL connection string (optional)
        """
        self.db_connection_string = db_connection_string
        self._engine: Optional[Engine] = None
        
        if db_connection_string and SQLALCHEMY_AVAILABLE:
            self._engine = create_engine(db_connection_string)
    
    def load_csv(
        self,
        filepath: str,
        parse_dates: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            parse_dates: List of columns to parse as dates
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_csv(filepath, parse_dates=parse_dates, **kwargs)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def load_all_csv(
        self,
        directory: str,
        parse_dates_map: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from a directory.
        
        Args:
            directory: Directory path
            parse_dates_map: Dict mapping filename to date columns
            
        Returns:
            Dictionary of DataFrames keyed by filename (without extension)
        """
        data = {}
        parse_dates_map = parse_dates_map or {}
        
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.warning(f"Directory {directory} does not exist")
            return data
        
        for filepath in directory_path.glob("*.csv"):
            name = filepath.stem
            parse_dates = parse_dates_map.get(name)
            data[name] = self.load_csv(str(filepath), parse_dates=parse_dates)
        
        logger.info(f"Loaded {len(data)} CSV files from {directory}")
        return data
    
    def load_parquet(
        self,
        filepath: str,
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Load data from Parquet file.
        
        Args:
            filepath: Path to Parquet file
            columns: List of columns to load (None for all)
            **kwargs: Additional arguments for pd.read_parquet
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {filepath}")
        
        df = pd.read_parquet(filepath, columns=columns, **kwargs)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def load_from_db(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Load data from PostgreSQL database.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        if not self._engine:
            raise ValueError("Database connection not configured")
        
        logger.info(f"Executing query: {query[:100]}...")
        df = pd.read_sql(query, self._engine, params=params)
        logger.info(f"Loaded {len(df)} rows from database")
        
        return df
    
    def load_table(self, table_name: str) -> pd.DataFrame:
        """
        Load entire table from database.
        
        Args:
            table_name: Name of the table
            
        Returns:
            DataFrame with table data
        """
        return self.load_from_db(f"SELECT * FROM {table_name}")
    
    def validate_data(
        self,
        df: pd.DataFrame,
        required_columns: List[str],
        date_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Validate DataFrame structure and content.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            date_columns: List of columns that should be dates
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValueError: If validation fails
        """
        # Check required columns
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert date columns
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
        
        # Check for empty DataFrame
        if len(df) == 0:
            logger.warning("DataFrame is empty")
        
        # Log basic stats
        logger.info(f"Validation passed: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def sample_data(
        self,
        df: pd.DataFrame,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Sample data for testing/development.
        
        Args:
            df: DataFrame to sample
            n: Number of rows to sample
            frac: Fraction of rows to sample
            random_state: Random seed
            
        Returns:
            Sampled DataFrame
        """
        if n is not None:
            sampled = df.sample(n=min(n, len(df)), random_state=random_state)
        elif frac is not None:
            sampled = df.sample(frac=frac, random_state=random_state)
        else:
            sampled = df
        
        logger.info(f"Sampled {len(sampled)} rows from {len(df)} total")
        return sampled


class DataPreparer:
    """
    Prepare data for analysis and modeling.
    
    Handles:
    - Missing values
    - Data type conversions
    - Feature scaling
    - Train/test splits
    """
    
    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = "drop",
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.
        
        Args:
            df: DataFrame with potential missing values
            strategy: 'drop', 'fill', 'mean', 'median', 'mode'
            fill_value: Value for 'fill' strategy
            columns: Columns to apply strategy to (None for all)
            
        Returns:
            DataFrame with handled missing values
        """
        df = df.copy()
        target_cols = columns or df.columns.tolist()
        
        initial_nulls = df[target_cols].isnull().sum().sum()
        
        if strategy == "drop":
            df = df.dropna(subset=target_cols)
        elif strategy == "fill":
            df[target_cols] = df[target_cols].fillna(fill_value)
        elif strategy == "mean":
            numeric_cols = df[target_cols].select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif strategy == "median":
            numeric_cols = df[target_cols].select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif strategy == "mode":
            for col in target_cols:
                df[col] = df[col].fillna(df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None)
        
        final_nulls = df[target_cols].isnull().sum().sum()
        logger.info(f"Missing values: {initial_nulls} -> {final_nulls}")
        
        return df
    
    @staticmethod
    def create_train_test_split(
        df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_column: Optional[str] = None
    ) -> tuple:
        """
        Split data into train and test sets.
        
        Args:
            df: DataFrame to split
            test_size: Proportion for test set
            random_state: Random seed
            stratify_column: Column for stratified split
            
        Returns:
            Tuple of (train_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        stratify = df[stratify_column] if stratify_column else None
        
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        logger.info(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        
        return train_df, test_df


# Data package init
__all__ = ['DataLoader', 'DataPreparer']
