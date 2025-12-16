"""
IC Analysis Module
==================

Comprehensive Information Coefficient (IC) analysis for factor evaluation:
- IC calculation (Spearman/Pearson)
- IC decay analysis
- IR (Information Ratio) computation
- IC significance testing
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class ICAnalyzer:
    """
    Information Coefficient (IC) analyzer for factor evaluation.
    
    IC measures the cross-sectional correlation between factor values
    and subsequent returns.
    """
    
    def __init__(
        self,
        method: str = 'spearman',
        forward_periods: int = 1,
        min_observations: int = 30
    ):
        """
        Initialize IC Analyzer.
        
        Parameters
        ----------
        method : str
            Correlation method: 'spearman' (rank) or 'pearson'
        forward_periods : int
            Number of periods for forward return calculation
        min_observations : int
            Minimum observations required per period
        """
        self.method = method
        self.forward_periods = forward_periods
        self.min_observations = min_observations
        self.ic_series = None
        self.ic_stats = None
    
    def compute_ic(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        factor_col: Optional[str] = None,
        return_col: Optional[str] = None
    ) -> pd.Series:
        """
        Compute time series of IC values.
        
        Parameters
        ----------
        factor_data : pd.DataFrame
            Factor values with MultiIndex (date, stock_id) or wide format
        returns : pd.DataFrame
            Return data in same format as factor_data
        factor_col : str, optional
            Column name if factor_data has multiple columns
        return_col : str, optional
            Column name if returns has multiple columns
            
        Returns
        -------
        pd.Series
            Time series of IC values indexed by date
        """
        # Prepare data
        if isinstance(factor_data.index, pd.MultiIndex):
            # MultiIndex format (date, stock_id)
            factor = factor_data[factor_col] if factor_col else factor_data.iloc[:, 0]
            ret = returns[return_col] if return_col else returns.iloc[:, 0]
            
            # Shift returns forward
            ret_shifted = ret.groupby(level=1).shift(-self.forward_periods)
            
            # Combine and compute IC per date
            combined = pd.DataFrame({
                'factor': factor,
                'return': ret_shifted
            }).dropna()
            
            ic_values = combined.groupby(level=0).apply(
                lambda x: self._compute_correlation(x['factor'], x['return'])
                if len(x) >= self.min_observations else np.nan
            )
            
        else:
            # Wide format (dates as index, stocks as columns)
            factor = factor_data[factor_col] if factor_col else factor_data
            ret = returns[return_col] if return_col else returns
            
            # Shift returns forward
            ret_shifted = ret.shift(-self.forward_periods)
            
            # Compute IC for each date
            ic_values = []
            for date in factor.index:
                if date in ret_shifted.index:
                    f = factor.loc[date]
                    r = ret_shifted.loc[date]
                    
                    # Remove NaN
                    valid = ~(f.isna() | r.isna())
                    if valid.sum() >= self.min_observations:
                        ic = self._compute_correlation(f[valid], r[valid])
                    else:
                        ic = np.nan
                    ic_values.append({'date': date, 'ic': ic})
            
            ic_values = pd.DataFrame(ic_values).set_index('date')['ic']
        
        self.ic_series = ic_values
        return ic_values
    
    def _compute_correlation(
        self, 
        x: pd.Series, 
        y: pd.Series
    ) -> float:
        """Compute correlation between two series."""
        if self.method == 'spearman':
            corr, _ = stats.spearmanr(x, y, nan_policy='omit')
        else:
            corr, _ = stats.pearsonr(x.dropna(), y.dropna())
        return corr
    
    def compute_ic_stats(
        self,
        ic_series: Optional[pd.Series] = None
    ) -> Dict:
        """
        Compute comprehensive IC statistics.
        
        Parameters
        ----------
        ic_series : pd.Series, optional
            IC time series (uses stored series if not provided)
            
        Returns
        -------
        dict
            Dictionary with IC statistics
        """
        if ic_series is None:
            ic_series = self.ic_series
        
        if ic_series is None:
            raise ValueError("No IC series available. Run compute_ic first.")
        
        ic_clean = ic_series.dropna()
        
        if len(ic_clean) == 0:
            return {}
        
        # Basic statistics
        ic_mean = ic_clean.mean()
        ic_std = ic_clean.std()
        ir = ic_mean / ic_std if ic_std > 0 else 0  # Information Ratio
        
        # T-statistic for IC mean
        t_stat = ic_mean / (ic_std / np.sqrt(len(ic_clean)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ic_clean) - 1))
        
        # IC decay (autocorrelation)
        ic_autocorr = ic_clean.autocorr(lag=1)
        
        # Positive IC ratio
        positive_ratio = (ic_clean > 0).mean()
        
        # Annualized IR (assuming daily data)
        ir_annual = ir * np.sqrt(252)
        
        self.ic_stats = {
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'ir': ir,
            'ir_annual': ir_annual,
            't_stat': t_stat,
            'p_value': p_value,
            'positive_ratio': positive_ratio,
            'ic_autocorr': ic_autocorr,
            'n_periods': len(ic_clean),
            'ic_median': ic_clean.median(),
            'ic_skew': ic_clean.skew(),
            'ic_kurtosis': ic_clean.kurtosis(),
            'ic_min': ic_clean.min(),
            'ic_max': ic_clean.max()
        }
        
        return self.ic_stats
    
    def compute_ic_decay(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        max_lag: int = 20,
        factor_col: Optional[str] = None,
        return_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute IC at different forward return lags.
        
        Parameters
        ----------
        factor_data : pd.DataFrame
            Factor values
        returns : pd.DataFrame
            Return data
        max_lag : int
            Maximum lag to compute
        factor_col : str, optional
            Factor column name
        return_col : str, optional
            Return column name
            
        Returns
        -------
        pd.DataFrame
            IC statistics at each lag
        """
        decay_results = []
        
        original_periods = self.forward_periods
        
        for lag in range(1, max_lag + 1):
            self.forward_periods = lag
            ic_series = self.compute_ic(factor_data, returns, factor_col, return_col)
            stats_dict = self.compute_ic_stats(ic_series)
            stats_dict['lag'] = lag
            decay_results.append(stats_dict)
        
        self.forward_periods = original_periods
        
        return pd.DataFrame(decay_results).set_index('lag')
    
    def compute_rolling_ic(
        self,
        ic_series: Optional[pd.Series] = None,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Compute rolling IC statistics.
        
        Parameters
        ----------
        ic_series : pd.Series, optional
            IC time series
        window : int
            Rolling window size
            
        Returns
        -------
        pd.DataFrame
            Rolling IC mean, std, and IR
        """
        if ic_series is None:
            ic_series = self.ic_series
        
        rolling_mean = ic_series.rolling(window, min_periods=window//2).mean()
        rolling_std = ic_series.rolling(window, min_periods=window//2).std()
        rolling_ir = rolling_mean / rolling_std
        
        return pd.DataFrame({
            'ic_mean': rolling_mean,
            'ic_std': rolling_std,
            'ir': rolling_ir
        })
    
    def compute_group_ic(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        group_col: str,
        factor_col: Optional[str] = None,
        return_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute IC within different groups (e.g., industries, size groups).
        
        Parameters
        ----------
        factor_data : pd.DataFrame
            Factor values with group column
        returns : pd.DataFrame
            Return data
        group_col : str
            Column name for group labels
        factor_col : str, optional
            Factor column name
        return_col : str, optional
            Return column name
            
        Returns
        -------
        pd.DataFrame
            IC statistics by group
        """
        groups = factor_data[group_col].unique()
        group_stats = []
        
        for group in groups:
            mask = factor_data[group_col] == group
            group_factor = factor_data[mask]
            group_returns = returns[mask]
            
            try:
                ic_series = self.compute_ic(group_factor, group_returns, 
                                           factor_col, return_col)
                stats_dict = self.compute_ic_stats(ic_series)
                stats_dict['group'] = group
                group_stats.append(stats_dict)
            except Exception:
                continue
        
        return pd.DataFrame(group_stats).set_index('group')
    
    def plot_ic_analysis(
        self,
        ic_series: Optional[pd.Series] = None,
        rolling_window: int = 60,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive IC analysis plots.
        
        Parameters
        ----------
        ic_series : pd.Series, optional
            IC time series
        rolling_window : int
            Window for rolling statistics
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        if ic_series is None:
            ic_series = self.ic_series
        
        if ic_series is None:
            raise ValueError("No IC series available")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. IC Time Series
        ax1 = axes[0, 0]
        ic_series.plot(ax=ax1, alpha=0.6, label='Daily IC')
        ic_series.rolling(rolling_window).mean().plot(
            ax=ax1, linewidth=2, label=f'{rolling_window}-day MA'
        )
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('IC Time Series')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('IC')
        ax1.legend()
        
        # 2. IC Distribution
        ax2 = axes[0, 1]
        ic_series.dropna().hist(bins=50, ax=ax2, alpha=0.7, edgecolor='black')
        ax2.axvline(ic_series.mean(), color='red', linestyle='--', 
                   label=f'Mean: {ic_series.mean():.4f}')
        ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('IC Distribution')
        ax2.set_xlabel('IC')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Cumulative IC
        ax3 = axes[1, 0]
        cumulative_ic = ic_series.cumsum()
        cumulative_ic.plot(ax=ax3)
        ax3.set_title('Cumulative IC')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative IC')
        
        # 4. Rolling IR
        ax4 = axes[1, 1]
        rolling_stats = self.compute_rolling_ic(ic_series, rolling_window)
        rolling_stats['ir'].plot(ax=ax4)
        ax4.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='IR=0.5')
        ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title(f'Rolling IR ({rolling_window}-day window)')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('IR')
        ax4.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def plot_ic_decay(
        self,
        decay_df: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 6),
        save_path: Optional[str] = None
    ):
        """
        Plot IC decay analysis.
        
        Parameters
        ----------
        decay_df : pd.DataFrame
            Output from compute_ic_decay()
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.bar(decay_df.index, decay_df['ic_mean'], alpha=0.7, 
               edgecolor='black', label='IC Mean')
        ax.errorbar(decay_df.index, decay_df['ic_mean'], 
                   yerr=decay_df['ic_std'], fmt='none', color='black', 
                   capsize=3, alpha=0.5)
        
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Forward Return Lag (periods)')
        ax.set_ylabel('IC Mean')
        ax.set_title('IC Decay Analysis')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def generate_report(
        self,
        factor_name: str = "Factor",
        ic_series: Optional[pd.Series] = None
    ) -> str:
        """
        Generate a text report of IC analysis.
        
        Parameters
        ----------
        factor_name : str
            Name of the factor for the report
        ic_series : pd.Series, optional
            IC time series
            
        Returns
        -------
        str
            Formatted report string
        """
        stats = self.compute_ic_stats(ic_series)
        
        if not stats:
            return "No IC statistics available."
        
        report = f"""
================================================================================
                        IC Analysis Report: {factor_name}
================================================================================

Summary Statistics:
------------------
  IC Mean:              {stats['ic_mean']:.4f}
  IC Std:               {stats['ic_std']:.4f}
  IC Median:            {stats['ic_median']:.4f}
  
Information Ratio:
-----------------
  IR (Daily):           {stats['ir']:.4f}
  IR (Annualized):      {stats['ir_annual']:.4f}
  
Statistical Significance:
------------------------
  T-Statistic:          {stats['t_stat']:.2f}
  P-Value:              {stats['p_value']:.4f}
  Significant (5%):     {'Yes' if stats['p_value'] < 0.05 else 'No'}
  
Additional Metrics:
------------------
  Positive IC Ratio:    {stats['positive_ratio']:.2%}
  IC Autocorrelation:   {stats['ic_autocorr']:.4f}
  IC Skewness:          {stats['ic_skew']:.4f}
  IC Kurtosis:          {stats['ic_kurtosis']:.4f}
  
  IC Range:             [{stats['ic_min']:.4f}, {stats['ic_max']:.4f}]
  Number of Periods:    {stats['n_periods']}

Interpretation Guidelines:
-------------------------
  IC Mean > 0.02:       Potentially meaningful factor
  IR > 0.3:             Good factor (daily)
  IR > 0.5:             Strong factor (daily)
  Positive Ratio > 55%: Consistent predictive power

================================================================================
"""
        return report
    
    def is_significant(
        self,
        min_ic: float = 0.02,
        min_ir: float = 0.3,
        max_p_value: float = 0.05
    ) -> bool:
        """
        Check if factor passes significance thresholds.
        
        Parameters
        ----------
        min_ic : float
            Minimum absolute IC mean
        min_ir : float
            Minimum IR
        max_p_value : float
            Maximum p-value for significance
            
        Returns
        -------
        bool
            True if factor is significant
        """
        if self.ic_stats is None:
            self.compute_ic_stats()
        
        if not self.ic_stats:
            return False
        
        return (
            abs(self.ic_stats['ic_mean']) >= min_ic and
            abs(self.ic_stats['ir']) >= min_ir and
            self.ic_stats['p_value'] <= max_p_value
        )

