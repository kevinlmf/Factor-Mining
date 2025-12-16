"""
Fundamental Factor Builder
==========================

Constructs fundamental/value factors from financial data:
- Valuation factors (B/M, E/P, etc.)
- Profitability factors (ROE, ROA, margins)
- Growth factors
- Quality factors
- Investment factors
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from .base import BaseFactor


class FundamentalFactorBuilder(BaseFactor):
    """
    Builder for fundamental/value-based factors.
    
    Implements classic and modern fundamental factors.
    """
    
    def __init__(self, name: str = "fundamental"):
        """
        Initialize Fundamental Factor Builder.
        
        Parameters
        ----------
        name : str
            Factor name
        """
        super().__init__(name=name, category="fundamental")
    
    def compute(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute all fundamental factors from financial data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Financial data with accounting items
            Expected columns vary by factor
            
        Returns
        -------
        pd.DataFrame
            All fundamental factors
        """
        result = pd.DataFrame(index=data.index)
        
        # Valuation factors
        if all(col in data.columns for col in ['book_value', 'market_cap']):
            result['book_to_market'] = self.book_to_market(data)
        
        if all(col in data.columns for col in ['earnings', 'market_cap']):
            result['earnings_yield'] = self.earnings_yield(data)
        
        if all(col in data.columns for col in ['sales', 'market_cap']):
            result['sales_to_price'] = self.sales_to_price(data)
        
        # Profitability factors
        if all(col in data.columns for col in ['net_income', 'equity']):
            result['roe'] = self.roe(data)
        
        if all(col in data.columns for col in ['net_income', 'total_assets']):
            result['roa'] = self.roa(data)
        
        if all(col in data.columns for col in ['gross_profit', 'total_assets']):
            result['gross_profitability'] = self.gross_profitability(data)
        
        # Quality factors
        if all(col in data.columns for col in ['operating_cash_flow', 'net_income']):
            result['accruals'] = self.accruals(data)
        
        # Growth factors
        if 'total_assets' in data.columns:
            result['asset_growth'] = self.asset_growth(data)
        
        if 'sales' in data.columns:
            result['sales_growth'] = self.sales_growth(data)
        
        # Investment factors
        if 'total_assets' in data.columns:
            result['investment'] = self.investment(data)
        
        self.factor_data = result
        return result
    
    # =====================
    # Valuation Factors
    # =====================
    
    def book_to_market(
        self,
        data: pd.DataFrame,
        book_col: str = 'book_value',
        market_col: str = 'market_cap'
    ) -> pd.Series:
        """
        Compute Book-to-Market ratio (HML factor basis).
        
        B/M = Book Value of Equity / Market Capitalization
        
        Parameters
        ----------
        data : pd.DataFrame
            Financial data
        book_col : str
            Book value column
        market_col : str
            Market cap column
            
        Returns
        -------
        pd.Series
            Book-to-market ratio
        """
        bm = data[book_col] / (data[market_col] + 1e-8)
        # Handle negative book values
        bm = bm.where(data[book_col] > 0, np.nan)
        return bm
    
    def earnings_yield(
        self,
        data: pd.DataFrame,
        earnings_col: str = 'earnings',
        market_col: str = 'market_cap'
    ) -> pd.Series:
        """
        Compute Earnings Yield (E/P).
        
        E/P = Earnings / Market Capitalization
        
        Parameters
        ----------
        data : pd.DataFrame
            Financial data
        earnings_col : str
            Earnings column
        market_col : str
            Market cap column
            
        Returns
        -------
        pd.Series
            Earnings yield
        """
        return data[earnings_col] / (data[market_col] + 1e-8)
    
    def sales_to_price(
        self,
        data: pd.DataFrame,
        sales_col: str = 'sales',
        market_col: str = 'market_cap'
    ) -> pd.Series:
        """
        Compute Sales-to-Price ratio.
        
        S/P = Sales / Market Capitalization
        
        Returns
        -------
        pd.Series
            Sales-to-price ratio
        """
        sp = data[sales_col] / (data[market_col] + 1e-8)
        return sp.where(data[sales_col] > 0, np.nan)
    
    # =====================
    # Profitability Factors
    # =====================
    
    def roe(
        self,
        data: pd.DataFrame,
        income_col: str = 'net_income',
        equity_col: str = 'equity'
    ) -> pd.Series:
        """
        Compute Return on Equity (ROE).
        
        ROE = Net Income / Shareholders' Equity
        
        Returns
        -------
        pd.Series
            ROE
        """
        roe = data[income_col] / (data[equity_col] + 1e-8)
        return roe.where(data[equity_col] > 0, np.nan)
    
    def roa(
        self,
        data: pd.DataFrame,
        income_col: str = 'net_income',
        assets_col: str = 'total_assets'
    ) -> pd.Series:
        """
        Compute Return on Assets (ROA).
        
        ROA = Net Income / Total Assets
        
        Returns
        -------
        pd.Series
            ROA
        """
        return data[income_col] / (data[assets_col] + 1e-8)
    
    def gross_profitability(
        self,
        data: pd.DataFrame,
        gross_profit_col: str = 'gross_profit',
        assets_col: str = 'total_assets'
    ) -> pd.Series:
        """
        Compute Gross Profitability (Novy-Marx, 2013).
        
        GP/A = Gross Profit / Total Assets
        
        Returns
        -------
        pd.Series
            Gross profitability
        """
        return data[gross_profit_col] / (data[assets_col] + 1e-8)
    
    def operating_profitability(
        self,
        data: pd.DataFrame
    ) -> pd.Series:
        """
        Compute Operating Profitability (Fama-French RMW basis).
        
        OP = (Revenue - COGS - SG&A - Interest) / Book Equity
        
        Returns
        -------
        pd.Series
            Operating profitability
        """
        required_cols = ['revenue', 'cogs', 'sga', 'interest_expense', 'equity']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(np.nan, index=data.index)
        
        operating_income = (
            data['revenue'] - data['cogs'] - 
            data.get('sga', 0) - data.get('interest_expense', 0)
        )
        return operating_income / (data['equity'] + 1e-8)
    
    # =====================
    # Quality Factors
    # =====================
    
    def accruals(
        self,
        data: pd.DataFrame,
        ocf_col: str = 'operating_cash_flow',
        income_col: str = 'net_income',
        assets_col: str = 'total_assets'
    ) -> pd.Series:
        """
        Compute Accruals (earnings quality measure).
        
        Accruals = (Net Income - Operating Cash Flow) / Total Assets
        
        Lower accruals indicate higher earnings quality.
        
        Returns
        -------
        pd.Series
            Accruals ratio (negated for factor alignment)
        """
        accruals = (data[income_col] - data[ocf_col]) / (data[assets_col] + 1e-8)
        # Negate so higher = better quality
        return -accruals
    
    def earnings_variability(
        self,
        data: pd.DataFrame,
        income_col: str = 'net_income',
        assets_col: str = 'total_assets',
        window: int = 20  # quarters
    ) -> pd.Series:
        """
        Compute earnings variability (stability measure).
        
        Returns
        -------
        pd.Series
            Earnings variability (negated - lower is better)
        """
        earnings_scaled = data[income_col] / (data[assets_col] + 1e-8)
        
        variability = data.groupby('stock_id').apply(
            lambda x: earnings_scaled.loc[x.index].rolling(window, min_periods=4).std()
        )
        
        if isinstance(variability, pd.DataFrame):
            variability = variability.stack()
        
        # Negate so higher = more stable
        return -variability
    
    # =====================
    # Growth Factors
    # =====================
    
    def asset_growth(
        self,
        data: pd.DataFrame,
        assets_col: str = 'total_assets',
        periods: int = 4  # quarters or years
    ) -> pd.Series:
        """
        Compute Asset Growth (Cooper et al., 2008).
        
        AG = (Total Assets_t - Total Assets_{t-1}) / Total Assets_{t-1}
        
        Returns
        -------
        pd.Series
            Asset growth rate
        """
        def calc_growth(group):
            return group[assets_col].pct_change(periods)
        
        result = data.groupby('stock_id').apply(calc_growth)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result
    
    def sales_growth(
        self,
        data: pd.DataFrame,
        sales_col: str = 'sales',
        periods: int = 4
    ) -> pd.Series:
        """
        Compute Sales Growth.
        
        Returns
        -------
        pd.Series
            Sales growth rate
        """
        def calc_growth(group):
            return group[sales_col].pct_change(periods)
        
        result = data.groupby('stock_id').apply(calc_growth)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result
    
    def earnings_growth(
        self,
        data: pd.DataFrame,
        earnings_col: str = 'earnings',
        periods: int = 4
    ) -> pd.Series:
        """
        Compute Earnings Growth.
        
        Returns
        -------
        pd.Series
            Earnings growth rate
        """
        def calc_growth(group):
            prev = group[earnings_col].shift(periods)
            # Use absolute value in denominator for negative earnings
            return (group[earnings_col] - prev) / (prev.abs() + 1e-8)
        
        result = data.groupby('stock_id').apply(calc_growth)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result
    
    # =====================
    # Investment Factors
    # =====================
    
    def investment(
        self,
        data: pd.DataFrame,
        assets_col: str = 'total_assets',
        periods: int = 4
    ) -> pd.Series:
        """
        Compute Investment factor (Fama-French CMA basis).
        
        Conservative investment = lower asset growth
        
        Returns
        -------
        pd.Series
            Investment factor (negated asset growth)
        """
        # Negate asset growth: conservative = positive
        return -self.asset_growth(data, assets_col, periods)
    
    def capex_to_assets(
        self,
        data: pd.DataFrame,
        capex_col: str = 'capex',
        assets_col: str = 'total_assets'
    ) -> pd.Series:
        """
        Compute Capital Expenditure to Assets ratio.
        
        Returns
        -------
        pd.Series
            CapEx / Total Assets
        """
        if capex_col not in data.columns:
            return pd.Series(np.nan, index=data.index)
        
        return data[capex_col] / (data[assets_col] + 1e-8)
    
    # =====================
    # Composite Quality Factor
    # =====================
    
    def quality_composite(
        self,
        data: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.Series:
        """
        Compute composite quality factor (QMJ-style).
        
        Combines profitability, safety, and growth metrics.
        
        Parameters
        ----------
        data : pd.DataFrame
            Financial data
        weights : dict, optional
            Weights for component factors
            
        Returns
        -------
        pd.Series
            Composite quality score
        """
        if weights is None:
            weights = {
                'roe': 0.25,
                'roa': 0.20,
                'accruals': 0.20,
                'asset_growth': 0.15,
                'gross_profitability': 0.20
            }
        
        # Compute component factors
        components = {}
        
        if all(col in data.columns for col in ['net_income', 'equity']):
            components['roe'] = self.roe(data)
        
        if all(col in data.columns for col in ['net_income', 'total_assets']):
            components['roa'] = self.roa(data)
        
        if all(col in data.columns for col in ['operating_cash_flow', 'net_income', 'total_assets']):
            components['accruals'] = self.accruals(data)
        
        if 'total_assets' in data.columns:
            components['asset_growth'] = -self.asset_growth(data)  # Negate: lower growth = higher quality
        
        if all(col in data.columns for col in ['gross_profit', 'total_assets']):
            components['gross_profitability'] = self.gross_profitability(data)
        
        # Standardize and combine
        if not components:
            return pd.Series(np.nan, index=data.index)
        
        # Z-score standardization per cross-section
        standardized = {}
        for name, series in components.items():
            grouped = series.groupby(level=0)  # Group by date
            standardized[name] = (series - grouped.transform('mean')) / grouped.transform('std')
        
        # Weighted combination
        composite = pd.Series(0.0, index=data.index)
        total_weight = 0
        
        for name, series in standardized.items():
            if name in weights:
                composite += weights[name] * series.fillna(0)
                total_weight += weights[name]
        
        if total_weight > 0:
            composite /= total_weight
        
        return composite

