"""
Microstructure Factor Builder
=============================

Constructs trading/microstructure factors:
- Volume-based factors
- Liquidity measures
- Price impact metrics
- Order flow indicators
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from .base import BaseFactor


class MicrostructureFactorBuilder(BaseFactor):
    """
    Builder for microstructure/trading-based factors.
    
    Implements various liquidity and market microstructure measures.
    """
    
    def __init__(self, name: str = "microstructure"):
        """
        Initialize Microstructure Factor Builder.
        
        Parameters
        ----------
        name : str
            Factor name
        """
        super().__init__(name=name, category="microstructure")
    
    def compute(
        self, 
        data: pd.DataFrame,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute all microstructure factors from market data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with columns: date, stock_id, open, high, low, close, 
            volume, amount, turnover (optional)
            
        Returns
        -------
        pd.DataFrame
            All microstructure factors
        """
        result = pd.DataFrame(index=data.index)
        
        # Compute individual factors
        if all(col in data.columns for col in ['close', 'volume', 'amount']):
            result['amihud_illiquidity'] = self.amihud_illiquidity(data)
        
        if 'volume' in data.columns:
            result['abnormal_volume'] = self.abnormal_volume(data)
            result['volume_momentum'] = self.volume_momentum(data)
        
        if 'turnover' in data.columns:
            result['turnover_volatility'] = self.turnover_volatility(data)
        
        if all(col in data.columns for col in ['high', 'low']):
            result['realized_volatility'] = self.realized_volatility(data)
            result['high_low_spread'] = self.high_low_spread(data)
        
        if all(col in data.columns for col in ['close', 'volume']):
            result['volume_price_trend'] = self.volume_price_trend(data)
        
        self.factor_data = result
        return result
    
    def amihud_illiquidity(
        self,
        data: pd.DataFrame,
        window: int = 20,
        price_col: str = 'close',
        volume_col: str = 'amount'  # Use amount (price * volume) for Amihud
    ) -> pd.Series:
        """
        Compute Amihud (2002) illiquidity measure.
        
        ILLIQ = avg(|r| / volume) * 10^6
        
        Higher values indicate less liquid stocks.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data
        window : int
            Rolling window for averaging
        price_col : str
            Price column name
        volume_col : str
            Trading amount column name
            
        Returns
        -------
        pd.Series
            Amihud illiquidity measure
        """
        # Calculate absolute returns
        returns = data.groupby('stock_id')[price_col].pct_change().abs()
        amount = data[volume_col]
        
        # Daily illiquidity
        daily_illiq = (returns / (amount + 1e-8)) * 1e6
        
        # Rolling average
        illiquidity = data.groupby('stock_id').apply(
            lambda x: daily_illiq.loc[x.index].rolling(window, min_periods=5).mean()
        )
        
        if isinstance(illiquidity, pd.DataFrame):
            illiquidity = illiquidity.stack()
        
        return illiquidity
    
    def abnormal_volume(
        self,
        data: pd.DataFrame,
        short_window: int = 5,
        long_window: int = 60,
        volume_col: str = 'volume'
    ) -> pd.Series:
        """
        Compute abnormal volume ratio.
        
        Abnormal Volume = short-term avg volume / long-term avg volume
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data
        short_window : int
            Short-term window
        long_window : int
            Long-term window
        volume_col : str
            Volume column name
            
        Returns
        -------
        pd.Series
            Abnormal volume ratio
        """
        def calc_abnormal_vol(group):
            short_avg = group[volume_col].rolling(short_window, min_periods=1).mean()
            long_avg = group[volume_col].rolling(long_window, min_periods=10).mean()
            return short_avg / (long_avg + 1e-8)
        
        result = data.groupby('stock_id').apply(calc_abnormal_vol)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result
    
    def volume_momentum(
        self,
        data: pd.DataFrame,
        window: int = 20,
        volume_col: str = 'volume'
    ) -> pd.Series:
        """
        Compute volume momentum (rate of change in volume).
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data
        window : int
            Lookback window
        volume_col : str
            Volume column name
            
        Returns
        -------
        pd.Series
            Volume momentum
        """
        def calc_vol_mom(group):
            current = group[volume_col].rolling(5, min_periods=1).mean()
            past = group[volume_col].shift(window).rolling(5, min_periods=1).mean()
            return (current - past) / (past + 1e-8)
        
        result = data.groupby('stock_id').apply(calc_vol_mom)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result
    
    def turnover_volatility(
        self,
        data: pd.DataFrame,
        window: int = 20,
        turnover_col: str = 'turnover'
    ) -> pd.Series:
        """
        Compute turnover volatility.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data
        window : int
            Rolling window
        turnover_col : str
            Turnover column name
            
        Returns
        -------
        pd.Series
            Turnover volatility
        """
        def calc_turnover_vol(group):
            return group[turnover_col].rolling(window, min_periods=5).std()
        
        result = data.groupby('stock_id').apply(calc_turnover_vol)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result
    
    def realized_volatility(
        self,
        data: pd.DataFrame,
        window: int = 20,
        method: str = 'parkinson'
    ) -> pd.Series:
        """
        Compute realized volatility using various estimators.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with high, low, close columns
        window : int
            Rolling window
        method : str
            Volatility estimator: 'close', 'parkinson', 'garman_klass'
            
        Returns
        -------
        pd.Series
            Realized volatility
        """
        def calc_vol(group):
            if method == 'close':
                # Close-to-close volatility
                returns = np.log(group['close'] / group['close'].shift(1))
                return returns.rolling(window, min_periods=5).std() * np.sqrt(252)
            
            elif method == 'parkinson':
                # Parkinson volatility (uses high-low range)
                log_hl = np.log(group['high'] / group['low'])
                parkinson_var = (1 / (4 * np.log(2))) * (log_hl ** 2)
                return np.sqrt(parkinson_var.rolling(window, min_periods=5).mean() * 252)
            
            elif method == 'garman_klass':
                # Garman-Klass volatility
                log_hl = np.log(group['high'] / group['low'])
                log_co = np.log(group['close'] / group['open'])
                gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
                return np.sqrt(gk_var.rolling(window, min_periods=5).mean() * 252)
            
            else:
                raise ValueError(f"Unknown volatility method: {method}")
        
        result = data.groupby('stock_id').apply(calc_vol)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result
    
    def high_low_spread(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        Compute high-low spread as liquidity proxy.
        
        Spread = 2 * (High - Low) / (High + Low)
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data
        window : int
            Rolling window for averaging
            
        Returns
        -------
        pd.Series
            Average high-low spread
        """
        def calc_spread(group):
            daily_spread = 2 * (group['high'] - group['low']) / (group['high'] + group['low'])
            return daily_spread.rolling(window, min_periods=5).mean()
        
        result = data.groupby('stock_id').apply(calc_spread)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result
    
    def volume_price_trend(
        self,
        data: pd.DataFrame,
        window: int = 20
    ) -> pd.Series:
        """
        Compute Volume-Price Trend (VPT) indicator.
        
        VPT measures the relationship between price changes and volume.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data
        window : int
            Rolling window
            
        Returns
        -------
        pd.Series
            VPT indicator
        """
        def calc_vpt(group):
            price_change = group['close'].pct_change()
            vpt = (price_change * group['volume']).cumsum()
            # Normalize by rolling mean
            vpt_norm = vpt / vpt.rolling(window, min_periods=5).mean()
            return vpt_norm
        
        result = data.groupby('stock_id').apply(calc_vpt)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result
    
    def kyle_lambda(
        self,
        data: pd.DataFrame,
        window: int = 20,
        price_col: str = 'close',
        volume_col: str = 'amount'
    ) -> pd.Series:
        """
        Compute Kyle's Lambda (price impact coefficient).
        
        Measures how much prices move per unit of trading volume.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data
        window : int
            Rolling regression window
        price_col : str
            Price column
        volume_col : str
            Volume/amount column
            
        Returns
        -------
        pd.Series
            Kyle's lambda
        """
        from scipy import stats
        
        def calc_lambda(group):
            returns = group[price_col].pct_change()
            signed_volume = np.sign(returns) * group[volume_col]
            
            lambdas = []
            for i in range(window, len(group)):
                ret_window = returns.iloc[i-window:i].values
                vol_window = signed_volume.iloc[i-window:i].values
                
                # Filter valid observations
                valid = ~(np.isnan(ret_window) | np.isnan(vol_window))
                if valid.sum() > window // 2:
                    slope, _, _, _, _ = stats.linregress(
                        vol_window[valid], ret_window[valid]
                    )
                    lambdas.append(abs(slope) * 1e6)
                else:
                    lambdas.append(np.nan)
            
            result = pd.Series([np.nan] * window + lambdas, index=group.index)
            return result
        
        result = data.groupby('stock_id').apply(calc_lambda)
        
        if isinstance(result, pd.DataFrame):
            result = result.stack()
            
        return result

