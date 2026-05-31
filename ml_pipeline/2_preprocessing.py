"""
Data Preprocessing Module
Handles missing values, outlier detection, normalization, and data cleaning
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Preprocess Tesla deliveries dataset"""
    
    def __init__(self, df):
        """
        Initialize preprocessor
        
        Args:
            df (pd.DataFrame): Raw dataframe
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.original_df = df.copy()
        
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values
        
        Args:
            strategy (str): 'mean', 'median', 'forward_fill', 'drop'
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        missing_before = self.df.isnull().sum().sum()
        
        if strategy == 'mean':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        elif strategy == 'forward_fill':
            self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        elif strategy == 'drop':
            self.df = self.df.dropna()
        
        missing_after = self.df.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return self.df
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        logger.info("Removing duplicate rows...")
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)
        logger.info(f"Rows: {before} -> {after} (Removed: {before - after})")
        return self.df
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        Detect outliers using IQR or Z-score
        
        Args:
            method (str): 'iqr' or 'zscore'
            threshold (float): IQR multiplier or Z-score threshold
        
        Returns:
            pd.DataFrame: Outlier mask
        """
        logger.info(f"Detecting outliers using {method} method...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_mask = pd.DataFrame(False, index=self.df.index, columns=numeric_cols)
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outlier_mask[col] = (self.df[col] < lower) | (self.df[col] > upper)
        
        elif method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outlier_mask[col] = np.abs(stats.zscore(self.df[col])) > threshold
        
        return outlier_mask
    
    def handle_outliers(self, method='clip', percentile=95):
        """
        Handle outliers
        
        Args:
            method (str): 'clip', 'remove', 'transform'
            percentile (float): Percentile for clipping
        """
        logger.info(f"Handling outliers using {method} method...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'clip':
            for col in numeric_cols:
                lower = self.df[col].quantile(0.01)
                upper = self.df[col].quantile(0.99)
                self.df[col] = self.df[col].clip(lower, upper)
        
        elif method == 'remove':
            outlier_mask = self.detect_outliers(method='iqr')
            before = len(self.df)
            self.df = self.df[~outlier_mask.any(axis=1)]
            after = len(self.df)
            logger.info(f"Removed {before - after} outlier rows")
        
        return self.df
    
    def encode_categorical(self):
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return self.df
    
    def normalize_features(self, method='standard'):
        """
        Normalize numerical features
        
        Args:
            method (str): 'standard' (Z-score) or 'minmax'
        """
        logger.info(f"Normalizing features using {method} method...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'standard':
            self.df[numeric_cols] = self.scaler.fit_transform(self.df[numeric_cols])
        
        elif method == 'minmax':
            scaler = MinMaxScaler()
            self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        
        logger.info(f"Normalized {len(numeric_cols)} numerical features")
        return self.df
    
    def get_preprocessed_data(self):
        """Return preprocessed dataframe"""
        return self.df
    
    def get_processing_summary(self):
        """Get preprocessing summary"""
        summary = {
            'original_shape': self.original_df.shape,
            'processed_shape': self.df.shape,
            'rows_removed': self.original_df.shape[0] - self.df.shape[0],
            'missing_values_original': self.original_df.isnull().sum().sum(),
            'missing_values_processed': self.df.isnull().sum().sum()
        }
        return summary


if __name__ == "__main__":
    from 1_data_loading import DataLoader
    
    # Example usage
    loader = DataLoader('tesla_deliveries_dataset_2015_2025.csv')
    df = loader.load_data()
    
    preprocessor = DataPreprocessor(df)
    preprocessor.handle_missing_values('mean')
    preprocessor.remove_duplicates()
    preprocessor.handle_outliers('clip')
    preprocessor.encode_categorical()
    preprocessor.normalize_features('standard')
    
    processed_df = preprocessor.get_preprocessed_data()
    logger.info(f"\nProcessing Summary: {preprocessor.get_processing_summary()}")
