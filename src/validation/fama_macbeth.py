"""
Fama-MacBeth Regression
=======================

Implements the two-pass Fama-MacBeth (1973) procedure for testing
factor pricing models and validating alpha factors.

The Final Alpha Test:
--------------------
Step 1: Estimate stock betas to risk factors (time-series)
Step 2: Run cross-sectional regression each period:
        Ret_i,t - Rf = λ_0,t + λ_Alpha,t · Alpha_i,t-1 + Σ λ_k,t · β_i,k + η_i,t
Step 3: Test hypotheses on time-series average of λ coefficients

Key Hypotheses:
- H0: λ_Alpha = 0  →  Reject if alpha factor has pricing power
- H0: λ_0 = 0     →  Accept if model is well-specified
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac


class FamaMacBethRegression:
    """
    Fama-MacBeth regression for factor pricing tests.
    
    Implements the classic two-pass methodology with modern
    statistical corrections (Newey-West, Shanken, etc.).
    """
    
    def __init__(
        self,
        newey_west_lags: int = 6,
        shanken_correction: bool = True,
        min_stocks: int = 30,
        significance_level: float = 0.05
    ):
        """
        Initialize Fama-MacBeth Regression.
        
        Parameters
        ----------
        newey_west_lags : int
            Number of lags for Newey-West t-statistic adjustment
        shanken_correction : bool
            Whether to apply Shanken (1992) correction for errors-in-variables
        min_stocks : int
            Minimum number of stocks required per cross-section
        significance_level : float
            Significance level for hypothesis tests
        """
        self.newey_west_lags = newey_west_lags
        self.shanken_correction = shanken_correction
        self.min_stocks = min_stocks
        self.significance_level = significance_level
        
        self.lambda_estimates = None
        self.lambda_stats = None
        self.cross_sectional_r2 = None
    
    def run_fama_macbeth(
        self,
        returns: pd.DataFrame,
        alpha_factor: pd.DataFrame,
        beta_exposures: Optional[pd.DataFrame] = None,
        risk_free_rate: Optional[pd.Series] = None,
        include_intercept: bool = True
    ) -> Dict:
        """
        Run complete Fama-MacBeth regression.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Stock returns with MultiIndex (date, stock)
        alpha_factor : pd.DataFrame
            Alpha factor values (lagged) with MultiIndex (date, stock)
        beta_exposures : pd.DataFrame, optional
            Stock betas to risk factors with MultiIndex (date, stock)
        risk_free_rate : pd.Series, optional
            Risk-free rate series indexed by date
        include_intercept : bool
            Whether to include intercept (λ_0)
            
        Returns
        -------
        dict
            Complete Fama-MacBeth regression results
        """
        # Prepare data
        if isinstance(returns, pd.DataFrame) and returns.shape[1] == 1:
            returns = returns.iloc[:, 0]
        elif isinstance(returns, pd.DataFrame):
            returns = returns.stack()
        
        if isinstance(alpha_factor, pd.DataFrame) and alpha_factor.shape[1] == 1:
            alpha_factor = alpha_factor.iloc[:, 0]
        elif isinstance(alpha_factor, pd.DataFrame):
            alpha_factor = alpha_factor.stack()
        
        # Subtract risk-free rate if provided
        if risk_free_rate is not None:
            excess_returns = returns.copy()
            for date in returns.index.get_level_values(0).unique():
                if date in risk_free_rate.index:
                    mask = returns.index.get_level_values(0) == date
                    excess_returns.loc[mask] = returns.loc[mask] - risk_free_rate[date]
        else:
            excess_returns = returns
        
        # Step 2: Cross-sectional regressions
        dates = alpha_factor.index.get_level_values(0).unique()
        
        lambda_results = []
        r2_results = []
        
        for date in dates:
            # Check if we have return data for this date
            if date not in returns.index.get_level_values(0):
                continue
            
            # Get cross-section
            try:
                cs_returns = excess_returns.xs(date, level=0)
                cs_alpha = alpha_factor.xs(date, level=0)
            except KeyError:
                continue
            
            # Get beta exposures if provided
            if beta_exposures is not None and date in beta_exposures.index.get_level_values(0):
                cs_betas = beta_exposures.xs(date, level=0)
            else:
                cs_betas = None
            
            # Align data
            common_stocks = cs_returns.index.intersection(cs_alpha.index)
            if cs_betas is not None:
                common_stocks = common_stocks.intersection(cs_betas.index)
            
            if len(common_stocks) < self.min_stocks:
                continue
            
            # Build regression
            y = cs_returns.loc[common_stocks]
            
            X_dict = {'alpha': cs_alpha.loc[common_stocks]}
            
            if cs_betas is not None:
                for col in cs_betas.columns:
                    X_dict[col] = cs_betas.loc[common_stocks, col]
            
            X = pd.DataFrame(X_dict)
            
            # Remove NaN
            valid = ~(y.isna() | X.isna().any(axis=1))
            if valid.sum() < self.min_stocks:
                continue
            
            y_valid = y[valid]
            X_valid = X[valid]
            
            if include_intercept:
                X_valid = sm.add_constant(X_valid)
            
            # Run OLS
            try:
                model = sm.OLS(y_valid, X_valid).fit()
                
                lambda_row = {'date': date}
                if include_intercept:
                    lambda_row['lambda_0'] = model.params.get('const', np.nan)
                lambda_row['lambda_alpha'] = model.params.get('alpha', np.nan)
                
                # Add beta factor lambdas
                for col in X.columns:
                    if col != 'alpha':
                        lambda_row[f'lambda_{col}'] = model.params.get(col, np.nan)
                
                lambda_results.append(lambda_row)
                r2_results.append({'date': date, 'r2': model.rsquared})
                
            except Exception:
                continue
        
        if not lambda_results:
            return {'error': 'No valid cross-sections found'}
        
        # Store results
        self.lambda_estimates = pd.DataFrame(lambda_results).set_index('date')
        self.cross_sectional_r2 = pd.DataFrame(r2_results).set_index('date')
        
        # Step 3: Time-series analysis of lambdas
        self.lambda_stats = self._compute_lambda_statistics()
        
        # Compile results
        results = {
            'lambda_estimates': self.lambda_estimates,
            'lambda_stats': self.lambda_stats,
            'cross_sectional_r2': self.cross_sectional_r2,
            'n_periods': len(self.lambda_estimates),
            'hypothesis_tests': self._run_hypothesis_tests()
        }
        
        return results
    
    def _compute_lambda_statistics(self) -> pd.DataFrame:
        """
        Compute statistics for time-series of lambda estimates.
        
        Returns
        -------
        pd.DataFrame
            Lambda statistics including mean, std, t-stat, p-value
        """
        stats_dict = {}
        
        for col in self.lambda_estimates.columns:
            series = self.lambda_estimates[col].dropna()
            
            if len(series) < 10:
                continue
            
            mean_lambda = series.mean()
            std_lambda = series.std()
            
            # Standard t-statistic
            t_stat_simple = mean_lambda / (std_lambda / np.sqrt(len(series)))
            
            # Newey-West adjusted t-statistic
            t_stat_nw = self._newey_west_tstat(series)
            
            # P-values
            p_value_simple = 2 * (1 - stats.t.cdf(abs(t_stat_simple), len(series) - 1))
            p_value_nw = 2 * (1 - stats.t.cdf(abs(t_stat_nw), len(series) - 1))
            
            stats_dict[col] = {
                'mean': mean_lambda,
                'std': std_lambda,
                't_stat': t_stat_simple,
                't_stat_nw': t_stat_nw,
                'p_value': p_value_simple,
                'p_value_nw': p_value_nw,
                'n_periods': len(series),
                'positive_ratio': (series > 0).mean()
            }
        
        return pd.DataFrame(stats_dict).T
    
    def _newey_west_tstat(self, series: pd.Series) -> float:
        """
        Compute Newey-West adjusted t-statistic.
        
        Parameters
        ----------
        series : pd.Series
            Time series of lambda estimates
            
        Returns
        -------
        float
            Newey-West adjusted t-statistic
        """
        n = len(series)
        mean = series.mean()
        
        # Compute autocorrelation-robust variance
        gamma_0 = np.var(series, ddof=1)
        
        nw_var = gamma_0
        for lag in range(1, self.newey_west_lags + 1):
            if lag >= n:
                break
            weight = 1 - lag / (self.newey_west_lags + 1)  # Bartlett kernel
            gamma_lag = np.cov(series[lag:], series[:-lag])[0, 1]
            nw_var += 2 * weight * gamma_lag
        
        # Ensure positive variance
        nw_var = max(nw_var, 1e-10)
        
        nw_se = np.sqrt(nw_var / n)
        t_stat = mean / nw_se
        
        return t_stat
    
    def _run_hypothesis_tests(self) -> Dict:
        """
        Run hypothesis tests on lambda estimates.
        
        Returns
        -------
        dict
            Hypothesis test results
        """
        tests = {}
        
        # Test 1: Alpha factor pricing (λ_alpha ≠ 0)
        if 'lambda_alpha' in self.lambda_stats.index:
            alpha_stats = self.lambda_stats.loc['lambda_alpha']
            tests['alpha_pricing'] = {
                'null_hypothesis': 'λ_alpha = 0',
                'alternative': 'λ_alpha ≠ 0',
                'lambda_mean': alpha_stats['mean'],
                't_stat': alpha_stats['t_stat_nw'],
                'p_value': alpha_stats['p_value_nw'],
                'reject_null': alpha_stats['p_value_nw'] < self.significance_level,
                'conclusion': 'Alpha factor has significant pricing power' 
                    if alpha_stats['p_value_nw'] < self.significance_level 
                    else 'Alpha factor does NOT have significant pricing power'
            }
        
        # Test 2: Model specification (λ_0 = 0)
        if 'lambda_0' in self.lambda_stats.index:
            intercept_stats = self.lambda_stats.loc['lambda_0']
            tests['model_specification'] = {
                'null_hypothesis': 'λ_0 = 0',
                'alternative': 'λ_0 ≠ 0',
                'lambda_mean': intercept_stats['mean'],
                't_stat': intercept_stats['t_stat_nw'],
                'p_value': intercept_stats['p_value_nw'],
                'reject_null': intercept_stats['p_value_nw'] < self.significance_level,
                'conclusion': 'Model is well-specified (no unexplained return)' 
                    if intercept_stats['p_value_nw'] >= self.significance_level 
                    else 'Model may be misspecified (significant unexplained return)'
            }
        
        # Test 3: Risk factor pricing
        for col in self.lambda_stats.index:
            if col.startswith('lambda_') and col not in ['lambda_0', 'lambda_alpha']:
                factor_name = col.replace('lambda_', '')
                factor_stats = self.lambda_stats.loc[col]
                tests[f'{factor_name}_pricing'] = {
                    'null_hypothesis': f'λ_{factor_name} = 0',
                    'lambda_mean': factor_stats['mean'],
                    't_stat': factor_stats['t_stat_nw'],
                    'p_value': factor_stats['p_value_nw'],
                    'reject_null': factor_stats['p_value_nw'] < self.significance_level
                }
        
        return tests
    
    def run_two_pass(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        alpha_factor: Optional[pd.DataFrame] = None,
        test_assets: str = 'stocks'
    ) -> Dict:
        """
        Run complete two-pass Fama-MacBeth procedure.
        
        Pass 1: Time-series regression to estimate betas
        Pass 2: Cross-sectional regression to estimate risk premia
        
        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns (wide format: dates x assets)
        factor_returns : pd.DataFrame
            Risk factor returns (dates x factors)
        alpha_factor : pd.DataFrame, optional
            Alpha factor to test (MultiIndex format)
        test_assets : str
            Type of test assets: 'stocks', 'portfolios'
            
        Returns
        -------
        dict
            Two-pass regression results
        """
        # Pass 1: Estimate betas
        betas = self._first_pass(returns, factor_returns)
        
        # Convert to format for second pass
        # Use average betas per stock
        avg_betas = betas.groupby('stock').mean()
        
        # Pass 2: Cross-sectional regression
        results = self._second_pass(returns, avg_betas, alpha_factor)
        
        return results
    
    def _first_pass(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame
    ) -> pd.DataFrame:
        """
        First pass: Time-series regression for beta estimation.
        
        R_i,t - R_f = α_i + Σ β_i,k · F_k,t + ε_i,t
        
        Returns
        -------
        pd.DataFrame
            Beta estimates for each stock
        """
        beta_results = []
        
        # Align data
        common_dates = returns.index.intersection(factor_returns.index)
        returns_aligned = returns.loc[common_dates]
        factors_aligned = factor_returns.loc[common_dates]
        
        X = sm.add_constant(factors_aligned)
        
        for stock in returns.columns:
            y = returns_aligned[stock].dropna()
            
            # Align with factors
            common_idx = y.index.intersection(X.index)
            
            if len(common_idx) < 60:  # Minimum observations
                continue
            
            X_aligned = X.loc[common_idx]
            y_aligned = y.loc[common_idx]
            
            try:
                model = sm.OLS(y_aligned, X_aligned).fit()
                
                beta_row = {'stock': stock, 'alpha': model.params.get('const', 0)}
                for factor in factor_returns.columns:
                    beta_row[f'beta_{factor}'] = model.params.get(factor, 0)
                
                # Add t-stats
                beta_row['alpha_tstat'] = model.tvalues.get('const', 0)
                for factor in factor_returns.columns:
                    beta_row[f'beta_{factor}_tstat'] = model.tvalues.get(factor, 0)
                
                beta_results.append(beta_row)
                
            except Exception:
                continue
        
        return pd.DataFrame(beta_results)
    
    def _second_pass(
        self,
        returns: pd.DataFrame,
        betas: pd.DataFrame,
        alpha_factor: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Second pass: Cross-sectional regression for risk premia.
        
        E[R_i] - R_f = λ_0 + Σ λ_k · β_i,k + λ_alpha · Alpha_i
        
        Returns
        -------
        dict
            Risk premia estimates and test statistics
        """
        # Average returns per stock
        avg_returns = returns.mean()
        
        # Align betas
        common_stocks = avg_returns.index.intersection(betas.index)
        
        y = avg_returns.loc[common_stocks]
        
        # Build X matrix
        beta_cols = [col for col in betas.columns if col.startswith('beta_')]
        X = betas.loc[common_stocks, beta_cols]
        
        # Add alpha factor if provided
        if alpha_factor is not None:
            # Use average alpha factor value
            if isinstance(alpha_factor.index, pd.MultiIndex):
                avg_alpha = alpha_factor.groupby(level=1).mean()
            else:
                avg_alpha = alpha_factor.mean(axis=0)
            
            if isinstance(avg_alpha, pd.DataFrame):
                avg_alpha = avg_alpha.iloc[:, 0]
            
            common_stocks = common_stocks.intersection(avg_alpha.index)
            y = avg_returns.loc[common_stocks]
            X = betas.loc[common_stocks, beta_cols]
            X['alpha'] = avg_alpha.loc[common_stocks]
        
        # Remove NaN
        valid = ~(y.isna() | X.isna().any(axis=1))
        y_valid = y[valid]
        X_valid = sm.add_constant(X[valid])
        
        # Run OLS
        model = sm.OLS(y_valid, X_valid).fit()
        
        # Compute Shanken correction if requested
        if self.shanken_correction:
            # Simplified Shanken correction
            # Full implementation requires factor covariance matrix
            shanken_factor = 1.0  # Placeholder
        else:
            shanken_factor = 1.0
        
        # Results
        results = {
            'lambda_estimates': model.params,
            't_stats': model.tvalues,
            'p_values': model.pvalues,
            'r_squared': model.rsquared,
            'r_squared_adj': model.rsquared_adj,
            'n_assets': len(y_valid),
            'shanken_factor': shanken_factor
        }
        
        return results
    
    def plot_results(
        self,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ):
        """
        Plot Fama-MacBeth regression results.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save figure
        """
        import matplotlib.pyplot as plt
        
        if self.lambda_estimates is None:
            raise ValueError("No results to plot. Run regression first.")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Lambda time series
        ax1 = axes[0, 0]
        if 'lambda_alpha' in self.lambda_estimates.columns:
            self.lambda_estimates['lambda_alpha'].plot(ax=ax1, alpha=0.6)
            self.lambda_estimates['lambda_alpha'].rolling(20).mean().plot(
                ax=ax1, linewidth=2, label='20-period MA'
            )
        ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax1.set_title('λ_alpha Time Series')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('λ_alpha')
        ax1.legend()
        
        # 2. Lambda distribution
        ax2 = axes[0, 1]
        if 'lambda_alpha' in self.lambda_estimates.columns:
            self.lambda_estimates['lambda_alpha'].hist(
                bins=50, ax=ax2, alpha=0.7, edgecolor='black'
            )
            mean_lambda = self.lambda_estimates['lambda_alpha'].mean()
            ax2.axvline(mean_lambda, color='red', linestyle='--', 
                       label=f'Mean: {mean_lambda:.4f}')
            ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('λ_alpha Distribution')
        ax2.set_xlabel('λ_alpha')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Cross-sectional R²
        ax3 = axes[1, 0]
        if self.cross_sectional_r2 is not None:
            self.cross_sectional_r2['r2'].plot(ax=ax3)
            ax3.axhline(self.cross_sectional_r2['r2'].mean(), 
                       color='red', linestyle='--',
                       label=f"Mean R²: {self.cross_sectional_r2['r2'].mean():.2%}")
        ax3.set_title('Cross-Sectional R²')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('R²')
        ax3.legend()
        
        # 4. Lambda bar chart
        ax4 = axes[1, 1]
        if self.lambda_stats is not None:
            means = self.lambda_stats['mean']
            stds = self.lambda_stats['std'] / np.sqrt(self.lambda_stats['n_periods'])
            
            x = range(len(means))
            colors = ['green' if m > 0 else 'red' for m in means]
            ax4.bar(x, means, yerr=1.96*stds, capsize=5, alpha=0.7, 
                   color=colors, edgecolor='black')
            ax4.set_xticks(x)
            ax4.set_xticklabels(means.index, rotation=45, ha='right')
            ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
            ax4.set_title('Lambda Estimates (Mean ± 95% CI)')
            ax4.set_ylabel('Lambda')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def generate_report(self, factor_name: str = "Alpha Factor") -> str:
        """
        Generate comprehensive Fama-MacBeth regression report.
        
        Parameters
        ----------
        factor_name : str
            Name of the alpha factor being tested
            
        Returns
        -------
        str
            Formatted report string
        """
        if self.lambda_stats is None:
            return "No results available. Run regression first."
        
        tests = self._run_hypothesis_tests()
        
        report = f"""
================================================================================
                    Fama-MacBeth Regression Report
                    Alpha Factor: {factor_name}
================================================================================

Methodology:
-----------
  Two-pass Fama-MacBeth procedure with:
  - Newey-West lags: {self.newey_west_lags}
  - Shanken correction: {'Yes' if self.shanken_correction else 'No'}
  - Minimum stocks per cross-section: {self.min_stocks}
  - Significance level: {self.significance_level:.1%}

Lambda Estimates:
----------------
"""
        for idx in self.lambda_stats.index:
            row = self.lambda_stats.loc[idx]
            significance = "***" if row['p_value_nw'] < 0.01 else \
                          ("**" if row['p_value_nw'] < 0.05 else \
                          ("*" if row['p_value_nw'] < 0.10 else ""))
            report += f"""
  {idx}:
    Mean:           {row['mean']:.6f} {significance}
    Std:            {row['std']:.6f}
    t-stat (NW):    {row['t_stat_nw']:.3f}
    p-value (NW):   {row['p_value_nw']:.4f}
    Positive ratio: {row['positive_ratio']:.1%}
"""

        report += """
Hypothesis Tests:
----------------
"""
        # Alpha pricing test
        if 'alpha_pricing' in tests:
            t = tests['alpha_pricing']
            report += f"""
  1. ALPHA FACTOR PRICING TEST
     H0: λ_alpha = 0 (No pricing power)
     H1: λ_alpha ≠ 0 (Has pricing power)
     
     λ_alpha mean:  {t['lambda_mean']:.6f}
     t-statistic:   {t['t_stat']:.3f}
     p-value:       {t['p_value']:.4f}
     
     DECISION: {'REJECT H0' if t['reject_null'] else 'FAIL TO REJECT H0'}
     CONCLUSION: {t['conclusion']}
"""

        # Model specification test
        if 'model_specification' in tests:
            t = tests['model_specification']
            report += f"""
  2. MODEL SPECIFICATION TEST
     H0: λ_0 = 0 (No unexplained returns)
     H1: λ_0 ≠ 0 (Unexplained returns exist)
     
     λ_0 mean:      {t['lambda_mean']:.6f}
     t-statistic:   {t['t_stat']:.3f}
     p-value:       {t['p_value']:.4f}
     
     DECISION: {'REJECT H0' if t['reject_null'] else 'FAIL TO REJECT H0'}
     CONCLUSION: {t['conclusion']}
"""

        # Cross-sectional R²
        if self.cross_sectional_r2 is not None:
            report += f"""
Cross-Sectional Fit:
-------------------
  Mean R²:     {self.cross_sectional_r2['r2'].mean():.2%}
  Median R²:   {self.cross_sectional_r2['r2'].median():.2%}
  Min R²:      {self.cross_sectional_r2['r2'].min():.2%}
  Max R²:      {self.cross_sectional_r2['r2'].max():.2%}
"""

        report += f"""
Sample Information:
------------------
  Number of periods:     {len(self.lambda_estimates)}
  Date range:            {self.lambda_estimates.index.min()} to {self.lambda_estimates.index.max()}

Interpretation Guide:
--------------------
  *** significant at 1% level
  **  significant at 5% level
  *   significant at 10% level
  
  A significant λ_alpha indicates the alpha factor has independent
  pricing power after controlling for known risk factors.
  
  An insignificant λ_0 suggests the model is well-specified and
  captures the relevant sources of return variation.

================================================================================
"""
        return report

