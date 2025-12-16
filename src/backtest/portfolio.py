"""
Portfolio Backtest Module
=========================

Quantile/group portfolio construction and backtesting:
- Quantile sorting
- Long-short portfolio construction
- Performance metrics
- Turnover analysis
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class PortfolioBacktest:
    """
    Portfolio backtester for factor evaluation.
    
    Constructs quantile portfolios based on factor values
    and evaluates performance.
    """
    
    def __init__(
        self,
        n_groups: int = 5,
        holding_period: int = 1,
        weight_method: str = 'equal',
        long_short: bool = True
    ):
        """
        Initialize Portfolio Backtester.
        
        Parameters
        ----------
        n_groups : int
            Number of quantile groups (e.g., 5 for quintiles)
        holding_period : int
            Holding period in trading days
        weight_method : str
            Portfolio weighting: 'equal', 'value', 'factor'
        long_short : bool
            Whether to compute long-short portfolio
        """
        self.n_groups = n_groups
        self.holding_period = holding_period
        self.weight_method = weight_method
        self.long_short = long_short
        
        self.group_returns = None
        self.portfolio_metrics = None
    
    def construct_portfolios(
        self,
        factor_data: pd.DataFrame,
        returns: pd.DataFrame,
        factor_col: Optional[str] = None,
        return_col: Optional[str] = None,
        market_cap: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Construct quantile portfolios based on factor values.
        
        Parameters
        ----------
        factor_data : pd.DataFrame
            Factor values with MultiIndex (date, stock_id) or wide format
        returns : pd.DataFrame
            Forward returns in same format
        factor_col : str, optional
            Factor column name
        return_col : str, optional
            Return column name
        market_cap : pd.DataFrame, optional
            Market cap for value-weighted portfolios
            
        Returns
        -------
        pd.DataFrame
            Returns for each quantile group and long-short
        """
        # Prepare data
        if isinstance(factor_data.index, pd.MultiIndex):
            factor = factor_data[factor_col] if factor_col else factor_data.iloc[:, 0]
            ret = returns[return_col] if return_col else returns.iloc[:, 0]
            
            # Forward returns
            ret_forward = ret.groupby(level=1).shift(-self.holding_period)
            
            # Combine
            combined = pd.DataFrame({
                'factor': factor,
                'return': ret_forward
            })
            
            if market_cap is not None:
                combined['market_cap'] = market_cap
            
        else:
            # Wide format - need to stack
            factor = factor_data.stack()
            factor.name = 'factor'
            ret = returns.shift(-self.holding_period).stack()
            ret.name = 'return'
            
            combined = pd.DataFrame({
                'factor': factor,
                'return': ret
            })
        
        # Assign quantile groups per date
        def assign_groups(group):
            valid = group['factor'].notna()
            group = group.copy()
            group['group'] = np.nan
            
            if valid.sum() >= self.n_groups:
                group.loc[valid, 'group'] = pd.qcut(
                    group.loc[valid, 'factor'],
                    self.n_groups,
                    labels=range(1, self.n_groups + 1),
                    duplicates='drop'
                )
            return group
        
        combined = combined.groupby(level=0).apply(assign_groups)
        
        # Fix index after groupby.apply
        if isinstance(combined.index, pd.MultiIndex) and len(combined.index.names) > 2:
            combined = combined.droplevel(0)
        
        # Compute group returns
        def compute_group_return(group):
            results = {}
            for g in range(1, self.n_groups + 1):
                mask = group['group'] == g
                if mask.sum() > 0:
                    if self.weight_method == 'equal':
                        results[f'G{g}'] = group.loc[mask, 'return'].mean()
                    elif self.weight_method == 'value' and 'market_cap' in group.columns:
                        weights = group.loc[mask, 'market_cap']
                        weights = weights / weights.sum()
                        results[f'G{g}'] = (group.loc[mask, 'return'] * weights).sum()
                    else:
                        results[f'G{g}'] = group.loc[mask, 'return'].mean()
                else:
                    results[f'G{g}'] = np.nan
            return pd.Series(results)
        
        group_returns = combined.groupby(level=0).apply(compute_group_return)
        
        # Add long-short portfolio
        if self.long_short:
            group_returns['L-S'] = group_returns[f'G{self.n_groups}'] - group_returns['G1']
        
        self.group_returns = group_returns
        return group_returns
    
    def compute_performance_metrics(
        self,
        group_returns: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        annualization_factor: int = 252
    ) -> pd.DataFrame:
        """
        Compute comprehensive performance metrics for each group.
        
        Parameters
        ----------
        group_returns : pd.DataFrame, optional
            Group returns (uses stored returns if not provided)
        risk_free_rate : float
            Annual risk-free rate
        annualization_factor : int
            Trading days per year
            
        Returns
        -------
        pd.DataFrame
            Performance metrics for each group
        """
        if group_returns is None:
            group_returns = self.group_returns
        
        if group_returns is None:
            raise ValueError("No group returns available")
        
        rf_daily = risk_free_rate / annualization_factor
        
        metrics = {}
        
        for col in group_returns.columns:
            ret = group_returns[col].dropna()
            
            if len(ret) == 0:
                continue
            
            # Basic returns
            mean_ret = ret.mean()
            total_ret = (1 + ret).prod() - 1
            annual_ret = (1 + mean_ret) ** annualization_factor - 1
            
            # Risk metrics
            vol = ret.std()
            annual_vol = vol * np.sqrt(annualization_factor)
            
            # Drawdown
            cumulative = (1 + ret).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Risk-adjusted returns
            excess_ret = ret - rf_daily
            sharpe = (excess_ret.mean() / excess_ret.std() * np.sqrt(annualization_factor)
                     if excess_ret.std() > 0 else 0)
            
            # Sortino ratio
            downside_ret = ret[ret < 0]
            downside_vol = downside_ret.std() * np.sqrt(annualization_factor) if len(downside_ret) > 0 else np.nan
            sortino = annual_ret / downside_vol if downside_vol > 0 else np.nan
            
            # Calmar ratio
            calmar = -annual_ret / max_drawdown if max_drawdown < 0 else np.nan
            
            # Win rate
            win_rate = (ret > 0).mean()
            
            # T-statistic
            t_stat = mean_ret / (vol / np.sqrt(len(ret))) if vol > 0 else 0
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(ret) - 1))
            
            metrics[col] = {
                'mean_daily_ret': mean_ret,
                'annual_return': annual_ret,
                'total_return': total_ret,
                'daily_volatility': vol,
                'annual_volatility': annual_vol,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'calmar_ratio': calmar,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                't_stat': t_stat,
                'p_value': p_value,
                'n_periods': len(ret)
            }
        
        self.portfolio_metrics = pd.DataFrame(metrics).T
        return self.portfolio_metrics
    
    def compute_turnover(
        self,
        factor_data: pd.DataFrame,
        factor_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute portfolio turnover for each quantile group.
        
        Parameters
        ----------
        factor_data : pd.DataFrame
            Factor values
        factor_col : str, optional
            Factor column name
            
        Returns
        -------
        pd.DataFrame
            Turnover statistics
        """
        if isinstance(factor_data.index, pd.MultiIndex):
            factor = factor_data[factor_col] if factor_col else factor_data.iloc[:, 0]
        else:
            factor = factor_data.stack()
        
        # Assign groups
        def assign_groups(group):
            valid = group.notna()
            result = pd.Series(np.nan, index=group.index)
            if valid.sum() >= self.n_groups:
                result[valid] = pd.qcut(
                    group[valid],
                    self.n_groups,
                    labels=range(1, self.n_groups + 1),
                    duplicates='drop'
                )
            return result
        
        groups = factor.groupby(level=0).apply(assign_groups)
        
        # Compute turnover per group
        turnover_results = {}
        
        for g in range(1, self.n_groups + 1):
            group_turnover = []
            dates = groups.index.get_level_values(0).unique().sort_values()
            
            for i in range(1, len(dates)):
                prev_date = dates[i-1]
                curr_date = dates[i]
                
                prev_stocks = set(groups.xs(prev_date)[groups.xs(prev_date) == g].index)
                curr_stocks = set(groups.xs(curr_date)[groups.xs(curr_date) == g].index)
                
                if len(prev_stocks) > 0 and len(curr_stocks) > 0:
                    # Turnover = % of stocks that changed
                    unchanged = len(prev_stocks & curr_stocks)
                    turnover = 1 - unchanged / max(len(prev_stocks), len(curr_stocks))
                    group_turnover.append(turnover)
            
            if group_turnover:
                turnover_results[f'G{g}'] = {
                    'mean_turnover': np.mean(group_turnover),
                    'median_turnover': np.median(group_turnover),
                    'turnover_std': np.std(group_turnover)
                }
        
        return pd.DataFrame(turnover_results).T
    
    def compute_monotonicity(
        self,
        group_returns: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Test for monotonic relationship between groups and returns.
        
        Parameters
        ----------
        group_returns : pd.DataFrame, optional
            Group returns
            
        Returns
        -------
        dict
            Monotonicity test results
        """
        if group_returns is None:
            group_returns = self.group_returns
        
        # Get group columns (exclude L-S)
        group_cols = [col for col in group_returns.columns if col.startswith('G')]
        
        # Mean returns per group
        mean_returns = group_returns[group_cols].mean()
        
        # Spearman rank correlation
        ranks = list(range(1, len(group_cols) + 1))
        return_ranks = mean_returns.rank()
        
        corr, p_value = stats.spearmanr(ranks, return_ranks)
        
        # Check strict monotonicity
        diffs = mean_returns.diff().dropna()
        is_monotonic_increasing = (diffs > 0).all()
        is_monotonic_decreasing = (diffs < 0).all()
        
        return {
            'spearman_correlation': corr,
            'p_value': p_value,
            'is_monotonic': is_monotonic_increasing or is_monotonic_decreasing,
            'direction': 'increasing' if is_monotonic_increasing else 
                        ('decreasing' if is_monotonic_decreasing else 'non-monotonic'),
            'mean_returns_by_group': mean_returns.to_dict()
        }
    
    def plot_performance(
        self,
        group_returns: Optional[pd.DataFrame] = None,
        figsize: Tuple[int, int] = (14, 12),
        save_path: Optional[str] = None
    ):
        """
        Create comprehensive performance plots.
        
        Parameters
        ----------
        group_returns : pd.DataFrame, optional
            Group returns
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        if group_returns is None:
            group_returns = self.group_returns
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Cumulative Returns
        ax1 = axes[0, 0]
        cumulative = (1 + group_returns).cumprod()
        cumulative.plot(ax=ax1)
        ax1.set_title('Cumulative Returns by Quantile')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend(loc='best')
        ax1.axhline(1, color='black', linestyle='--', alpha=0.5)
        
        # 2. Average Returns by Group (Bar Chart)
        ax2 = axes[0, 1]
        mean_returns = group_returns.mean() * 252  # Annualized
        colors = ['green' if r > 0 else 'red' for r in mean_returns]
        mean_returns.plot(kind='bar', ax=ax2, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Annualized Return by Quantile')
        ax2.set_xlabel('Quantile Group')
        ax2.set_ylabel('Annualized Return')
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=0)
        
        # 3. Long-Short Portfolio Performance
        ax3 = axes[1, 0]
        if 'L-S' in group_returns.columns:
            ls_cumulative = (1 + group_returns['L-S']).cumprod()
            ls_cumulative.plot(ax=ax3, linewidth=2)
            ax3.fill_between(ls_cumulative.index, 1, ls_cumulative, 
                           alpha=0.3, where=(ls_cumulative > 1))
            ax3.set_title('Long-Short Portfolio Cumulative Returns')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Cumulative Return')
            ax3.axhline(1, color='black', linestyle='--', alpha=0.5)
        
        # 4. Drawdown
        ax4 = axes[1, 1]
        if 'L-S' in group_returns.columns:
            ls_cumulative = (1 + group_returns['L-S']).cumprod()
            running_max = ls_cumulative.cummax()
            drawdown = (ls_cumulative - running_max) / running_max * 100
            drawdown.plot(ax=ax4, color='red', alpha=0.7)
            ax4.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            ax4.set_title('Long-Short Drawdown')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Drawdown (%)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def generate_report(
        self,
        factor_name: str = "Factor"
    ) -> str:
        """
        Generate a text report of portfolio backtest results.
        
        Parameters
        ----------
        factor_name : str
            Name of the factor
            
        Returns
        -------
        str
            Formatted report string
        """
        if self.portfolio_metrics is None:
            self.compute_performance_metrics()
        
        metrics = self.portfolio_metrics
        monotonicity = self.compute_monotonicity()
        
        report = f"""
================================================================================
                    Portfolio Backtest Report: {factor_name}
================================================================================

Portfolio Construction:
----------------------
  Number of Groups:     {self.n_groups}
  Holding Period:       {self.holding_period} day(s)
  Weighting Method:     {self.weight_method}

Performance by Quantile (Annualized):
------------------------------------
"""
        for idx in metrics.index:
            row = metrics.loc[idx]
            report += f"""
  {idx}:
    Return:     {row['annual_return']:.2%}
    Volatility: {row['annual_volatility']:.2%}
    Sharpe:     {row['sharpe_ratio']:.2f}
    Max DD:     {row['max_drawdown']:.2%}
    Win Rate:   {row['win_rate']:.2%}
"""

        if 'L-S' in metrics.index:
            ls = metrics.loc['L-S']
            report += f"""
Long-Short Portfolio Highlights:
-------------------------------
  Annualized Return:    {ls['annual_return']:.2%}
  Annualized Volatility:{ls['annual_volatility']:.2%}
  Sharpe Ratio:         {ls['sharpe_ratio']:.2f}
  Sortino Ratio:        {ls['sortino_ratio']:.2f}
  Maximum Drawdown:     {ls['max_drawdown']:.2%}
  T-Statistic:          {ls['t_stat']:.2f}
  P-Value:              {ls['p_value']:.4f}
"""

        report += f"""
Monotonicity Analysis:
---------------------
  Spearman Correlation: {monotonicity['spearman_correlation']:.4f}
  P-Value:              {monotonicity['p_value']:.4f}
  Is Monotonic:         {'Yes' if monotonicity['is_monotonic'] else 'No'}
  Direction:            {monotonicity['direction']}

================================================================================
"""
        return report

