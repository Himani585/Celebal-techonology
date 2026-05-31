"""
Data Loading Module
Handles data import, initial inspection, and basic statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Load and inspect Tesla deliveries dataset"""
    
    def __init__(self, file_path):
        """
        Initialize DataLoader
        
        Args:
            file_path (str): Path to CSV file
        """
        self.file_path = file_path
        self.df = None
        self.original_shape = None
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.df = pd.read_csv(self.file_path)
            self.original_shape = self.df.shape
            logger.info(f"Data loaded successfully. Shape: {self.original_shape}")
            return self.df
        except FileNotFoundError:
            logger.error(f"File not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_basic_info(self):
        """Get basic information about dataset"""
        if self.df is None:
            logger.error("Data not loaded. Call load_data() first.")
            return None
        
        info = {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        return info
    
    def display_statistics(self):
        """Display descriptive statistics"""
        if self.df is None:
            logger.error("Data not loaded.")
            return None
        
        logger.info("\n=== DATASET STATISTICS ===")
        logger.info(f"\nDataset Shape: {self.df.shape}")
        logger.info(f"\nFirst few rows:\n{self.df.head()}")
        logger.info(f"\nData Types:\n{self.df.dtypes}")
        logger.info(f"\nMissing Values:\n{self.df.isnull().sum()}")
        logger.info(f"\nDuplicate Rows: {self.df.duplicated().sum()}")
        logger.info(f"\nBasic Statistics:\n{self.df.describe()}")
        
        return self.df.describe()
    
    def check_data_quality(self):
        """Check data quality metrics"""
        if self.df is None:
            logger.error("Data not loaded.")
            return None
        
        quality_report = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_percentage': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))) * 100,
            'duplicate_rows': self.df.duplicated().sum(),
            'complete_rows': len(self.df.dropna()),
            'incomplete_rows': self.df.isnull().any(axis=1).sum()
        }
        
        logger.info("\n=== DATA QUALITY REPORT ===")
        for key, value in quality_report.items():
            logger.info(f"{key}: {value}")
        
        return quality_report
    
    def get_dataframe(self):
        """Return loaded dataframe"""
        return self.df


if __name__ == "__main__":
    # Example usage
    loader = DataLoader('tesla_deliveries_dataset_2015_2025.csv')
    df = loader.load_data()
    loader.display_statistics()
    loader.check_data_quality()
