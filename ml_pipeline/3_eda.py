"""
Exploratory Data Analysis (EDA) Module
Statistical analysis, visualizations, and correlation studies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis, pearsonr
import logging

logger = logging.getLogger(__name__)

class ExploratoryDataAnalysis:
    """Perform comprehensive EDA on Tesla deliveries dataset"""
    
    def __init__(self, df):
        """
        Initialize EDA
        
        Args:
            df (pd.DataFrame): Preprocessed dataframe
        """
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns
        
    def statistical_summary(self):
        """Generate statistical summary"""
        logger.info("=== STATISTICAL SUMMARY ===")
        
        summary = pd.DataFrame({
            'Count': self.df[self.numeric_cols].count(),
            'Mean': self.df[self.numeric_cols].mean(),
            'Std': self.df[self.numeric_cols].std(),
            'Min': self.df[self.numeric_cols].min(),
            '25%': self.df[self.numeric_cols].quantile(0.25),
            'Median': self.df[self.numeric_cols].median(),
            '75%': self.df[self.numeric_cols].quantile(0.75),
            'Max': self.df[self.numeric_cols].max(),
            'Skewness': self.df[self.numeric_cols].apply(skew),
            'Kurtosis': self.df[self.numeric_cols].apply(kurtosis)
        })
        
        logger.info(f"\n{summary}")
        return summary
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        logger.info("=== CORRELATION ANALYSIS ===")
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        logger.info(f"\nCorrelation Matrix:\n{corr_matrix}")
        
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
        
        if high_corr_pairs:
            logger.info(f"\nHigh Correlations (> 0.7):")
            for pair in high_corr_pairs:
                logger.info(f"{pair['Feature 1']} <-> {pair['Feature 2']}: {pair['Correlation']:.4f}")
        
        return corr_matrix, high_corr_pairs
    
    def distribution_analysis(self):
        """Analyze feature distributions"""
        logger.info("=== DISTRIBUTION ANALYSIS ===")
        
        distributions = {}
        for col in self.numeric_cols:
            distributions[col] = {
                'Skewness': skew(self.df[col].dropna()),
                'Kurtosis': kurtosis(self.df[col].dropna()),
                'Is_Normal': 'Yes' if abs(skew(self.df[col].dropna())) < 0.5 else 'No'
            }
            logger.info(f"\n{col}:")
            logger.info(f"  Skewness: {distributions[col]['Skewness']:.4f}")
            logger.info(f"  Kurtosis: {distributions[col]['Kurtosis']:.4f}")
            logger.info(f"  Normal Distribution: {distributions[col]['Is_Normal']}")
        
        return distributions
    
    def categorical_analysis(self):
        """Analyze categorical features"""
        logger.info("=== CATEGORICAL ANALYSIS ===")
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            logger.info(f"\n{col}:")
            logger.info(f"  Unique Values: {self.df[col].nunique()}")
            logger.info(f"  Value Counts:\n{self.df[col].value_counts()}")
    
    def target_variable_analysis(self, target_col='Estimated_Deliveries'):
        """Analyze target variable"""
        logger.info(f"\n=== TARGET VARIABLE ANALYSIS: {target_col} ===")
        
        if target_col in self.df.columns:
            logger.info(f"\nBasic Statistics:")
            logger.info(f"  Mean: {self.df[target_col].mean():.2f}")
            logger.info(f"  Median: {self.df[target_col].median():.2f}")
            logger.info(f"  Std Dev: {self.df[target_col].std():.2f}")
            logger.info(f"  Min: {self.df[target_col].min():.2f}")
            logger.info(f"  Max: {self.df[target_col].max():.2f}")
            logger.info(f"  Skewness: {skew(self.df[target_col]):.4f}")
            
            return self.df[target_col].describe()
        else:
            logger.warning(f"Target column '{target_col}' not found!")
            return None
    
    def missing_data_patterns(self):
        """Analyze missing data patterns"""
        logger.info("=== MISSING DATA ANALYSIS ===")
        
        missing_data = pd.DataFrame({
            'Column': self.df.columns,
            'Missing_Count': self.df.isnull().sum(),
            'Missing_Percentage': (self.df.isnull().sum() / len(self.df)) * 100
        })
        
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Percentage', ascending=False)
        
        if len(missing_data) > 0:
            logger.info(f"\n{missing_data}")
        else:
            logger.info("No missing values found!")
        
        return missing_data
    
    def feature_importance_preview(self):
        """Provide preview of potential important features"""
        logger.info("=== FEATURE IMPORTANCE PREVIEW ===")
        
        # Calculate correlation with target (assuming Estimated_Deliveries)
        target = 'Estimated_Deliveries'
        if target in self.df.columns:
            correlations = self.df[self.numeric_cols].corrwith(self.df[target]).abs().sort_values(ascending=False)
            logger.info(f"\nTop Features by Correlation with {target}:")
            logger.info(f"\n{correlations.head(10)}")
            return correlations
        else:
            logger.warning(f"Target column '{target}' not found!")
            return None
    
    def outlier_summary(self):
        """Summarize outliers in dataset"""
        logger.info("=== OUTLIER SUMMARY ===")
        
        outlier_counts = {}
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            
            outliers = len(self.df[(self.df[col] < lower) | (self.df[col] > upper)])
            outlier_counts[col] = {
                'Outlier_Count': outliers,
                'Outlier_Percentage': (outliers / len(self.df)) * 100,
                'Lower_Bound': lower,
                'Upper_Bound': upper
            }
        
        logger.info("\nOutliers by Column:")
        for col, stats in outlier_counts.items():
            if stats['Outlier_Count'] > 0:
                logger.info(f"{col}: {stats['Outlier_Count']} ({stats['Outlier_Percentage']:.2f}%)")
        
        return outlier_counts
    
    def generate_eda_report(self):
        """Generate comprehensive EDA report"""
        logger.info("\n" + "="*60)
        logger.info("COMPREHENSIVE EDA REPORT")
        logger.info("="*60)
        
        self.statistical_summary()
        self.correlation_analysis()
        self.distribution_analysis()
        self.categorical_analysis()
        self.target_variable_analysis()
        self.missing_data_patterns()
        self.feature_importance_preview()
        self.outlier_summary()
        
        logger.info("\n" + "="*60)
        logger.info("EDA COMPLETE")
        logger.info("="*60)


if __name__ == "__main__":
    from 1_data_loading import DataLoader
    from 2_preprocessing import DataPreprocessor
    
    # Example usage
    loader = DataLoader('tesla_deliveries_dataset_2015_2025.csv')
    df = loader.load_data()
    
    preprocessor = DataPreprocessor(df)
    preprocessor.handle_missing_values('mean')
    processed_df = preprocessor.get_preprocessed_data()
    
    eda = ExploratoryDataAnalysis(processed_df)
    eda.generate_eda_report()
