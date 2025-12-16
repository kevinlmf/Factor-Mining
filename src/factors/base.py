"""
Base Factor Class
=================

Provides the foundation for all factor implementations with:
- Standardization (z-score, rank)
- Winsorization
- Missing value handling
- Factor decay analysis
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict
from scipy import stats


class BaseFactor(ABC):
    """
    Abstract base class for all factors.
    
    Provides common preprocessing and analysis methods.
    """
    
    def __init__(self, name: str, category: str):
        """
        Initialize base factor.
        
        Parameters
        ----------
        name : str
            Factor name
        category : str
            Factor category (nlp, microstructure, fundamental)
        """
        self.name = name
        self.category = category
        self.factor_data = None
        self._is_preprocessed = False
    
    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Compute factor values from raw data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data with MultiIndex (date, stock_id)
            
        Returns
        -------
        pd.DataFrame
            Factor values with same index
        """
        pass
    
    def preprocess(
        self,
        factor_data: pd.DataFrame,
        winsorize: bool = True,
        winsorize_limits: tuple = (0.01, 0.99),
        standardize: str = 'zscore',
        fillna_method: str = 'median',
        industry_neutral: bool = False,
        industry_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Preprocess factor data with standardization and cleaning.
        
        Parameters
        ----------
        factor_data : pd.DataFrame
            Raw factor values, MultiIndex (date, stock_id)
        winsorize : bool
            Whether to winsorize extreme values
        winsorize_limits : tuple
            Percentile limits for winsorization
        standardize : str
            Standardization method: 'zscore', 'rank', 'minmax', or None
        fillna_method : str
            Missing value fill method: 'median', 'mean', 'zero', 'forward'
        industry_neutral : bool
            Whether to neutralize by industry
        industry_col : str, optional
            Industry column name if industry_neutral=True
            
        Returns
        -------
        pd.DataFrame
            Preprocessed factor data
        """
        result = factor_data.copy()
        
        # Group by date for cross-sectional operations
        for date in result.index.get_level_values(0).unique():
            mask = result.index.get_level_values(0) == date
            date_data = result.loc[mask].copy()
            
            # Fill missing values
            if fillna_method == 'median':
                date_data = date_data.fillna(date_data.median())
            elif fillna_method == 'mean':
                date_data = date_data.fillna(date_data.mean())
            elif fillna_method == 'zero':
                date_data = date_data.fillna(0)
            
            # Winsorize
            if winsorize:
                date_data = self._winsorize(date_data, winsorize_limits)
            
            # Industry neutralization
            if industry_neutral and industry_col is not None:
                date_data = self._industry_neutralize(date_data, industry_col)
            
            # Standardize
            if standardize == 'zscore':
                date_data = self._zscore(date_data)
            elif standardize == 'rank':
                date_data = self._rank_normalize(date_data)
            elif standardize == 'minmax':
                date_data = self._minmax_normalize(date_data)
            
            result.loc[mask] = date_data.values
        
        self.factor_data = result
        self._is_preprocessed = True
        
        return result
    
    @staticmethod
    def _winsorize(data: pd.DataFrame, limits: tuple) -> pd.DataFrame:
        """Winsorize data to specified percentile limits."""
        result = data.copy()
        for col in result.columns:
            lower = result[col].quantile(limits[0])
            upper = result[col].quantile(limits[1])
            result[col] = result[col].clip(lower=lower, upper=upper)
        return result
    
    @staticmethod
    def _zscore(data: pd.DataFrame) -> pd.DataFrame:
        """Z-score standardization."""
        return (data - data.mean()) / data.std()
    
    @staticmethod
    def _rank_normalize(data: pd.DataFrame) -> pd.DataFrame:
        """Rank normalization to [-1, 1]."""
        result = data.rank(pct=True) * 2 - 1
        return result
    
    @staticmethod
    def _minmax_normalize(data: pd.DataFrame) -> pd.DataFrame:
        """Min-max normalization to [0, 1]."""
        return (data - data.min()) / (data.max() - data.min())
    
    @staticmethod
    def _industry_neutralize(
        data: pd.DataFrame, 
        industry_col: str
    ) -> pd.DataFrame:
        """
        Industry neutralization via demeaning within industry groups.
        """
        result = data.copy()
        if industry_col in result.columns:
            for col in result.columns:
                if col != industry_col:
                    industry_means = result.groupby(industry_col)[col].transform('mean')
                    result[col] = result[col] - industry_means
        return result
    
    def compute_factor_decay(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        max_lag: int = 20
    ) -> pd.DataFrame:
        """
        Analyze factor decay by computing IC at different lags.
        
        Parameters
        ----------
        factor_data : pd.DataFrame
            Factor values
        returns : pd.DataFrame
            Forward returns
        max_lag : int
            Maximum lag to analyze
            
        Returns
        -------
        pd.DataFrame
            IC values at each lag
        """
        decay_results = []
        
        for lag in range(1, max_lag + 1):
            # Compute IC at this lag
            shifted_returns = returns.shift(-lag)
            valid_mask = ~(factor_data.isna() | shifted_returns.isna())
            
            if valid_mask.sum() > 0:
                ic = factor_data[valid_mask].corrwith(
                    shifted_returns[valid_mask],
                    method='spearman'
                ).mean()
                decay_results.append({'lag': lag, 'ic': ic})
        
        return pd.DataFrame(decay_results)
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics of the factor."""
        if self.factor_data is None:
            return {}
        
        return {
            'name': self.name,
            'category': self.category,
            'mean': self.factor_data.mean().mean(),
            'std': self.factor_data.std().mean(),
            'skew': self.factor_data.skew().mean(),
            'kurtosis': self.factor_data.kurtosis().mean(),
            'missing_pct': self.factor_data.isna().mean().mean() * 100,
            'n_observations': len(self.factor_data)
        }

