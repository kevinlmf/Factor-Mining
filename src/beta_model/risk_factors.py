"""
Risk Factor Model
=================

Constructs and manages systematic risk factors (Beta factors):
- Market (MKT)
- Size (SMB)
- Value (HML)
- Momentum (MOM)
- Profitability (RMW)
- Investment (CMA)
- Quality (QMJ)

These factors serve as Formula 1 in the factor research framework.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from scipy import stats
import statsmodels.api as sm


class RiskFactorModel:
    """
    Risk Factor Model for beta estimation and risk factor construction.
    
    Implements Fama-French style factor construction and provides
    methods for estimating stock betas to these factors.
    """
    
    def __init__(
        self,
        factors: Optional[List[str]] = None,
        lookback_window: int = 252,
        min_observations: int = 60
    ):
        """
        Initialize Risk Factor Model.
        
        Parameters
        ----------
        factors : List[str], optional
            List of risk factors to construct
            Default: ['MKT', 'SMB', 'HML', 'MOM']
        lookback_window : int
            Window for beta estimation
        min_observations : int
            Minimum observations required for beta estimation
        """
        self.factors = factors or ['MKT', 'SMB', 'HML', 'MOM']
        self.lookback_window = lookback_window
        self.min_observations = min_observations
        
        self.factor_returns = None
        self.stock_betas = None
    
    # =====================
    # Factor Construction
    # =====================
    
    def construct_factors(
        self,
        returns: pd.DataFrame,
        market_cap: pd.DataFrame,
        book_to_market: Optional[pd.DataFrame] = None,
        momentum: Optional[pd.DataFrame] = None,
        profitability: Optional[pd.DataFrame] = None,
        investment: Optional[pd.DataFrame] = None,
        risk_free_rate: Optional[pd.Series] = None,
        rebalance_freq: str = 'M'
    ) -> pd.DataFrame:
        """
        Construct Fama-French style risk factors.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Stock returns (wide format: dates x stocks)
        market_cap : pd.DataFrame
            Market capitalization (same format)
        book_to_market : pd.DataFrame, optional
            Book-to-market ratio for HML
        momentum : pd.DataFrame, optional
            Past returns for MOM factor
        profitability : pd.DataFrame, optional
            Operating profitability for RMW
        investment : pd.DataFrame, optional
            Asset growth for CMA
        risk_free_rate : pd.Series, optional
            Risk-free rate series
        rebalance_freq : str
            Rebalancing frequency: 'D', 'W', 'M', 'Q', 'Y'
            
        Returns
        -------
        pd.DataFrame
            Factor returns time series
        """
        factor_returns = {}
        
        # Market factor (MKT-RF)
        if 'MKT' in self.factors:
            mkt = self._construct_market_factor(returns, market_cap, risk_free_rate)
            factor_returns['MKT'] = mkt
        
        # Size factor (SMB)
        if 'SMB' in self.factors:
            smb = self._construct_size_factor(returns, market_cap, rebalance_freq)
            factor_returns['SMB'] = smb
        
        # Value factor (HML)
        if 'HML' in self.factors and book_to_market is not None:
            hml = self._construct_value_factor(returns, book_to_market, 
                                               market_cap, rebalance_freq)
            factor_returns['HML'] = hml
        
        # Momentum factor (MOM)
        if 'MOM' in self.factors:
            if momentum is None:
                momentum = self._compute_momentum(returns)
            mom = self._construct_momentum_factor(returns, momentum, 
                                                   market_cap, rebalance_freq)
            factor_returns['MOM'] = mom
        
        # Profitability factor (RMW)
        if 'RMW' in self.factors and profitability is not None:
            rmw = self._construct_profitability_factor(returns, profitability,
                                                        market_cap, rebalance_freq)
            factor_returns['RMW'] = rmw
        
        # Investment factor (CMA)
        if 'CMA' in self.factors and investment is not None:
            cma = self._construct_investment_factor(returns, investment,
                                                     market_cap, rebalance_freq)
            factor_returns['CMA'] = cma
        
        self.factor_returns = pd.DataFrame(factor_returns)
        return self.factor_returns
    
    def _construct_market_factor(
        self,
        returns: pd.DataFrame,
        market_cap: pd.DataFrame,
        risk_free_rate: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Construct market factor (value-weighted market return - rf).
        """
        # Value-weighted market return
        aligned_cap = market_cap.reindex_like(returns)
        weights = aligned_cap.div(aligned_cap.sum(axis=1), axis=0)
        market_return = (returns * weights).sum(axis=1)
        
        # Subtract risk-free rate
        if risk_free_rate is not None:
            rf = risk_free_rate.reindex(market_return.index).fillna(0)
            market_return = market_return - rf
        
        return market_return
    
    def _construct_size_factor(
        self,
        returns: pd.DataFrame,
        market_cap: pd.DataFrame,
        rebalance_freq: str = 'M'
    ) -> pd.Series:
        """
        Construct size factor (SMB = Small Minus Big).
        """
        # Get rebalance dates
        if rebalance_freq == 'D':
            rebalance_dates = returns.index
        else:
            rebalance_dates = returns.resample(rebalance_freq).last().index
        
        smb_returns = []
        
        prev_small = None
        prev_big = None
        prev_rebal_date = None
        
        for date in returns.index:
            # Check if rebalance
            if prev_rebal_date is None or date >= rebalance_dates[
                rebalance_dates > prev_rebal_date].min() if len(
                rebalance_dates[rebalance_dates > prev_rebal_date]) > 0 else False:
                
                if date in market_cap.index:
                    caps = market_cap.loc[date].dropna()
                    if len(caps) >= 2:
                        median_cap = caps.median()
                        prev_small = caps[caps <= median_cap].index.tolist()
                        prev_big = caps[caps > median_cap].index.tolist()
                        prev_rebal_date = date
            
            # Compute return
            if prev_small and prev_big and date in returns.index:
                day_returns = returns.loc[date]
                small_ret = day_returns[day_returns.index.isin(prev_small)].mean()
                big_ret = day_returns[day_returns.index.isin(prev_big)].mean()
                smb_returns.append({'date': date, 'return': small_ret - big_ret})
            else:
                smb_returns.append({'date': date, 'return': np.nan})
        
        result = pd.DataFrame(smb_returns).set_index('date')['return']
        return result
    
    def _construct_value_factor(
        self,
        returns: pd.DataFrame,
        book_to_market: pd.DataFrame,
        market_cap: pd.DataFrame,
        rebalance_freq: str = 'M'
    ) -> pd.Series:
        """
        Construct value factor (HML = High B/M Minus Low B/M).
        """
        hml_returns = []
        
        prev_high = None
        prev_low = None
        prev_rebal_date = None
        
        rebalance_dates = returns.resample(rebalance_freq).last().index
        
        for date in returns.index:
            # Check if rebalance
            if prev_rebal_date is None or date >= rebalance_dates[
                rebalance_dates > prev_rebal_date].min() if len(
                rebalance_dates[rebalance_dates > prev_rebal_date]) > 0 else False:
                
                if date in book_to_market.index:
                    bm = book_to_market.loc[date].dropna()
                    if len(bm) >= 3:
                        # Top 30% vs bottom 30%
                        high_thresh = bm.quantile(0.7)
                        low_thresh = bm.quantile(0.3)
                        prev_high = bm[bm >= high_thresh].index.tolist()
                        prev_low = bm[bm <= low_thresh].index.tolist()
                        prev_rebal_date = date
            
            # Compute return
            if prev_high and prev_low and date in returns.index:
                day_returns = returns.loc[date]
                high_ret = day_returns[day_returns.index.isin(prev_high)].mean()
                low_ret = day_returns[day_returns.index.isin(prev_low)].mean()
                hml_returns.append({'date': date, 'return': high_ret - low_ret})
            else:
                hml_returns.append({'date': date, 'return': np.nan})
        
        result = pd.DataFrame(hml_returns).set_index('date')['return']
        return result
    
    def _compute_momentum(
        self,
        returns: pd.DataFrame,
        lookback: int = 252,
        skip: int = 21
    ) -> pd.DataFrame:
        """
        Compute momentum signal (past 12-month return, skipping recent month).
        """
        # Rolling returns excluding recent month
        total_ret = (1 + returns).rolling(lookback, min_periods=lookback//2).apply(
            lambda x: np.prod(x) - 1, raw=True
        )
        recent_ret = (1 + returns).rolling(skip, min_periods=skip//2).apply(
            lambda x: np.prod(x) - 1, raw=True
        )
        
        momentum = (1 + total_ret) / (1 + recent_ret) - 1
        return momentum
    
    def _construct_momentum_factor(
        self,
        returns: pd.DataFrame,
        momentum: pd.DataFrame,
        market_cap: pd.DataFrame,
        rebalance_freq: str = 'M'
    ) -> pd.Series:
        """
        Construct momentum factor (Winners Minus Losers).
        """
        mom_returns = []
        
        prev_winners = None
        prev_losers = None
        prev_rebal_date = None
        
        rebalance_dates = returns.resample(rebalance_freq).last().index
        
        for date in returns.index:
            # Check if rebalance
            if prev_rebal_date is None or date >= rebalance_dates[
                rebalance_dates > prev_rebal_date].min() if len(
                rebalance_dates[rebalance_dates > prev_rebal_date]) > 0 else False:
                
                if date in momentum.index:
                    mom = momentum.loc[date].dropna()
                    if len(mom) >= 3:
                        high_thresh = mom.quantile(0.7)
                        low_thresh = mom.quantile(0.3)
                        prev_winners = mom[mom >= high_thresh].index.tolist()
                        prev_losers = mom[mom <= low_thresh].index.tolist()
                        prev_rebal_date = date
            
            # Compute return
            if prev_winners and prev_losers and date in returns.index:
                day_returns = returns.loc[date]
                winner_ret = day_returns[day_returns.index.isin(prev_winners)].mean()
                loser_ret = day_returns[day_returns.index.isin(prev_losers)].mean()
                mom_returns.append({'date': date, 'return': winner_ret - loser_ret})
            else:
                mom_returns.append({'date': date, 'return': np.nan})
        
        result = pd.DataFrame(mom_returns).set_index('date')['return']
        return result
    
    def _construct_profitability_factor(
        self,
        returns: pd.DataFrame,
        profitability: pd.DataFrame,
        market_cap: pd.DataFrame,
        rebalance_freq: str = 'M'
    ) -> pd.Series:
        """
        Construct profitability factor (RMW = Robust Minus Weak).
        """
        # Similar structure to HML
        rmw_returns = []
        
        prev_robust = None
        prev_weak = None
        prev_rebal_date = None
        
        rebalance_dates = returns.resample(rebalance_freq).last().index
        
        for date in returns.index:
            if prev_rebal_date is None or date >= rebalance_dates[
                rebalance_dates > prev_rebal_date].min() if len(
                rebalance_dates[rebalance_dates > prev_rebal_date]) > 0 else False:
                
                if date in profitability.index:
                    prof = profitability.loc[date].dropna()
                    if len(prof) >= 3:
                        high_thresh = prof.quantile(0.7)
                        low_thresh = prof.quantile(0.3)
                        prev_robust = prof[prof >= high_thresh].index.tolist()
                        prev_weak = prof[prof <= low_thresh].index.tolist()
                        prev_rebal_date = date
            
            if prev_robust and prev_weak and date in returns.index:
                day_returns = returns.loc[date]
                robust_ret = day_returns[day_returns.index.isin(prev_robust)].mean()
                weak_ret = day_returns[day_returns.index.isin(prev_weak)].mean()
                rmw_returns.append({'date': date, 'return': robust_ret - weak_ret})
            else:
                rmw_returns.append({'date': date, 'return': np.nan})
        
        result = pd.DataFrame(rmw_returns).set_index('date')['return']
        return result
    
    def _construct_investment_factor(
        self,
        returns: pd.DataFrame,
        investment: pd.DataFrame,
        market_cap: pd.DataFrame,
        rebalance_freq: str = 'M'
    ) -> pd.Series:
        """
        Construct investment factor (CMA = Conservative Minus Aggressive).
        """
        cma_returns = []
        
        prev_conservative = None
        prev_aggressive = None
        prev_rebal_date = None
        
        rebalance_dates = returns.resample(rebalance_freq).last().index
        
        for date in returns.index:
            if prev_rebal_date is None or date >= rebalance_dates[
                rebalance_dates > prev_rebal_date].min() if len(
                rebalance_dates[rebalance_dates > prev_rebal_date]) > 0 else False:
                
                if date in investment.index:
                    inv = investment.loc[date].dropna()
                    if len(inv) >= 3:
                        # Low investment = conservative
                        low_thresh = inv.quantile(0.3)
                        high_thresh = inv.quantile(0.7)
                        prev_conservative = inv[inv <= low_thresh].index.tolist()
                        prev_aggressive = inv[inv >= high_thresh].index.tolist()
                        prev_rebal_date = date
            
            if prev_conservative and prev_aggressive and date in returns.index:
                day_returns = returns.loc[date]
                cons_ret = day_returns[day_returns.index.isin(prev_conservative)].mean()
                aggr_ret = day_returns[day_returns.index.isin(prev_aggressive)].mean()
                cma_returns.append({'date': date, 'return': cons_ret - aggr_ret})
            else:
                cma_returns.append({'date': date, 'return': np.nan})
        
        result = pd.DataFrame(cma_returns).set_index('date')['return']
        return result
    
    # =====================
    # Beta Estimation
    # =====================
    
    def estimate_betas(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: Optional[pd.DataFrame] = None,
        method: str = 'rolling',
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Estimate stock betas to risk factors.
        
        Parameters
        ----------
        stock_returns : pd.DataFrame
            Stock returns (wide format)
        factor_returns : pd.DataFrame, optional
            Risk factor returns (uses constructed factors if not provided)
        method : str
            Estimation method: 'rolling', 'expanding', 'full_sample'
        window : int, optional
            Window size for rolling estimation
            
        Returns
        -------
        pd.DataFrame
            Beta estimates with MultiIndex (date, stock) if rolling,
            or single index (stock) if full_sample
        """
        if factor_returns is None:
            factor_returns = self.factor_returns
        
        if factor_returns is None:
            raise ValueError("No factor returns available")
        
        if window is None:
            window = self.lookback_window
        
        # Align dates
        common_dates = stock_returns.index.intersection(factor_returns.index)
        stock_returns = stock_returns.loc[common_dates]
        factor_returns = factor_returns.loc[common_dates]
        
        if method == 'full_sample':
            return self._estimate_betas_full_sample(stock_returns, factor_returns)
        elif method == 'rolling':
            return self._estimate_betas_rolling(stock_returns, factor_returns, window)
        elif method == 'expanding':
            return self._estimate_betas_expanding(stock_returns, factor_returns)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _estimate_betas_full_sample(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Estimate betas using full sample regression."""
        betas = {}
        
        X = sm.add_constant(factor_returns.dropna())
        
        for stock in stock_returns.columns:
            y = stock_returns[stock].dropna()
            
            # Align data
            common_idx = X.index.intersection(y.index)
            if len(common_idx) < self.min_observations:
                continue
            
            X_aligned = X.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            try:
                model = sm.OLS(y_aligned, X_aligned).fit()
                stock_betas = {'alpha': model.params.get('const', 0)}
                for factor in factor_returns.columns:
                    stock_betas[f'beta_{factor}'] = model.params.get(factor, 0)
                betas[stock] = stock_betas
            except Exception:
                continue
        
        return pd.DataFrame(betas).T
    
    def _estimate_betas_rolling(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        window: int
    ) -> pd.DataFrame:
        """Estimate betas using rolling window regression."""
        all_betas = []
        
        dates = stock_returns.index[window:]
        
        for date in dates:
            # Get window data
            window_start = stock_returns.index.get_loc(date) - window
            window_end = stock_returns.index.get_loc(date)
            
            window_returns = stock_returns.iloc[window_start:window_end]
            window_factors = factor_returns.iloc[window_start:window_end]
            
            X = sm.add_constant(window_factors.dropna())
            
            for stock in window_returns.columns:
                y = window_returns[stock].dropna()
                
                common_idx = X.index.intersection(y.index)
                if len(common_idx) < self.min_observations:
                    continue
                
                X_aligned = X.loc[common_idx]
                y_aligned = y.loc[common_idx]
                
                try:
                    model = sm.OLS(y_aligned, X_aligned).fit()
                    beta_row = {
                        'date': date,
                        'stock': stock,
                        'alpha': model.params.get('const', 0)
                    }
                    for factor in factor_returns.columns:
                        beta_row[f'beta_{factor}'] = model.params.get(factor, 0)
                    all_betas.append(beta_row)
                except Exception:
                    continue
        
        result = pd.DataFrame(all_betas)
        if len(result) > 0:
            result = result.set_index(['date', 'stock'])
        
        self.stock_betas = result
        return result
    
    def _estimate_betas_expanding(
        self,
        stock_returns: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """Estimate betas using expanding window."""
        all_betas = []
        
        min_window = self.min_observations
        dates = stock_returns.index[min_window:]
        
        for date in dates:
            window_end = stock_returns.index.get_loc(date)
            
            window_returns = stock_returns.iloc[:window_end]
            window_factors = factor_returns.iloc[:window_end]
            
            X = sm.add_constant(window_factors.dropna())
            
            for stock in window_returns.columns:
                y = window_returns[stock].dropna()
                
                common_idx = X.index.intersection(y.index)
                if len(common_idx) < self.min_observations:
                    continue
                
                X_aligned = X.loc[common_idx]
                y_aligned = y.loc[common_idx]
                
                try:
                    model = sm.OLS(y_aligned, X_aligned).fit()
                    beta_row = {
                        'date': date,
                        'stock': stock,
                        'alpha': model.params.get('const', 0)
                    }
                    for factor in factor_returns.columns:
                        beta_row[f'beta_{factor}'] = model.params.get(factor, 0)
                    all_betas.append(beta_row)
                except Exception:
                    continue
        
        result = pd.DataFrame(all_betas)
        if len(result) > 0:
            result = result.set_index(['date', 'stock'])
        
        return result
    
    def get_factor_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics for risk factors.
        
        Returns
        -------
        pd.DataFrame
            Factor statistics
        """
        if self.factor_returns is None:
            raise ValueError("No factor returns available")
        
        stats_dict = {}
        
        for col in self.factor_returns.columns:
            series = self.factor_returns[col].dropna()
            
            mean_ret = series.mean()
            std_ret = series.std()
            sharpe = mean_ret / std_ret * np.sqrt(252) if std_ret > 0 else 0
            
            # T-stat for mean
            t_stat = mean_ret / (std_ret / np.sqrt(len(series)))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(series) - 1))
            
            stats_dict[col] = {
                'mean_daily': mean_ret,
                'mean_annual': mean_ret * 252,
                'std_daily': std_ret,
                'std_annual': std_ret * np.sqrt(252),
                'sharpe': sharpe,
                't_stat': t_stat,
                'p_value': p_value,
                'n_observations': len(series)
            }
        
        return pd.DataFrame(stats_dict).T
    
    def get_factor_correlations(self) -> pd.DataFrame:
        """
        Compute correlation matrix of risk factors.
        
        Returns
        -------
        pd.DataFrame
            Correlation matrix
        """
        if self.factor_returns is None:
            raise ValueError("No factor returns available")
        
        return self.factor_returns.corr()

