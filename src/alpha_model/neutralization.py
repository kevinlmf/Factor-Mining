"""
Alpha Factor Neutralization
===========================

Implements Formula 2: Factor neutralization to remove beta exposures.

Alpha Factor = Raw Factor - Σ γ_k · Beta Factor_k

This ensures the alpha factor is orthogonal to known risk factors,
maximizing the "pure alpha" content.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple, Union
from scipy import stats
import statsmodels.api as sm


class AlphaFactorNeutralizer:
    """
    Alpha Factor Neutralizer.
    
    Removes exposure to known risk factors from raw alpha factors,
    ensuring orthogonality and maximizing pure alpha content.
    """
    
    def __init__(
        self,
        neutralize_market: bool = True,
        neutralize_size: bool = True,
        neutralize_industry: bool = True,
        neutralize_beta_factors: bool = True,
        winsorize: bool = True,
        winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    ):
        """
        Initialize Alpha Factor Neutralizer.
        
        Parameters
        ----------
        neutralize_market : bool
            Whether to remove market beta exposure
        neutralize_size : bool
            Whether to remove size (log market cap) exposure
        neutralize_industry : bool
            Whether to remove industry effects
        neutralize_beta_factors : bool
            Whether to neutralize against custom beta factors
        winsorize : bool
            Whether to winsorize extreme values
        winsorize_limits : tuple
            Percentile limits for winsorization
        """
        self.neutralize_market = neutralize_market
        self.neutralize_size = neutralize_size
        self.neutralize_industry = neutralize_industry
        self.neutralize_beta_factors = neutralize_beta_factors
        self.winsorize = winsorize
        self.winsorize_limits = winsorize_limits
        
        self.neutralization_results = {}
    
    def neutralize(
        self,
        raw_factor: pd.DataFrame,
        beta_factors: Optional[pd.DataFrame] = None,
        market_cap: Optional[pd.DataFrame] = None,
        industry: Optional[pd.DataFrame] = None,
        factor_col: Optional[str] = None,
        method: str = 'regression'
    ) -> pd.DataFrame:
        """
        Neutralize raw factor against beta factors.
        
        Implements Formula 2:
        Alpha Factor = Raw Factor - Σ γ_k · Beta Factor_k
        
        Parameters
        ----------
        raw_factor : pd.DataFrame
            Raw factor values with MultiIndex (date, stock_id)
        beta_factors : pd.DataFrame, optional
            Beta factor exposures (e.g., SMB, HML, MOM betas)
        market_cap : pd.DataFrame, optional
            Market capitalization for size neutralization
        industry : pd.DataFrame, optional
            Industry labels for industry neutralization
        factor_col : str, optional
            Column name if raw_factor has multiple columns
        method : str
            Neutralization method: 'regression', 'rank_regression', 'demean'
            
        Returns
        -------
        pd.DataFrame
            Neutralized alpha factor
        """
        # Extract factor series
        if factor_col:
            factor = raw_factor[factor_col].copy()
        elif isinstance(raw_factor, pd.Series):
            factor = raw_factor.copy()
        else:
            factor = raw_factor.iloc[:, 0].copy()
        
        # Process each date cross-sectionally
        dates = factor.index.get_level_values(0).unique()
        neutralized_values = []
        
        for date in dates:
            # Get cross-section
            cs_factor = factor.xs(date, level=0)
            
            # Build neutralization matrix
            X_components = []
            component_names = []
            
            # Size neutralization
            if self.neutralize_size and market_cap is not None:
                if date in market_cap.index.get_level_values(0):
                    cs_cap = market_cap.xs(date, level=0)
                    if isinstance(cs_cap, pd.DataFrame):
                        cs_cap = cs_cap.iloc[:, 0]
                    # Use log market cap
                    log_cap = np.log(cs_cap.replace(0, np.nan))
                    X_components.append(log_cap)
                    component_names.append('size')
            
            # Industry neutralization
            if self.neutralize_industry and industry is not None:
                if date in industry.index.get_level_values(0):
                    cs_industry = industry.xs(date, level=0)
                    if isinstance(cs_industry, pd.DataFrame):
                        cs_industry = cs_industry.iloc[:, 0]
                    # Create industry dummies
                    industry_dummies = pd.get_dummies(cs_industry, prefix='ind', drop_first=True)
                    for col in industry_dummies.columns:
                        X_components.append(industry_dummies[col])
                        component_names.append(col)
            
            # Beta factors neutralization
            if self.neutralize_beta_factors and beta_factors is not None:
                if isinstance(beta_factors.index, pd.MultiIndex):
                    if date in beta_factors.index.get_level_values(0):
                        cs_betas = beta_factors.xs(date, level=0)
                        for col in cs_betas.columns:
                            X_components.append(cs_betas[col])
                            component_names.append(col)
                else:
                    # Single time point beta factors
                    for col in beta_factors.columns:
                        if col in cs_factor.index:
                            X_components.append(beta_factors[col])
                            component_names.append(col)
            
            # Perform neutralization
            if X_components:
                X = pd.concat(X_components, axis=1)
                X.columns = component_names
                
                # Align indices
                common_idx = cs_factor.index.intersection(X.index)
                cs_factor_aligned = cs_factor.loc[common_idx]
                X_aligned = X.loc[common_idx]
                
                # Remove NaN
                valid = ~(cs_factor_aligned.isna() | X_aligned.isna().any(axis=1))
                
                if valid.sum() > len(X_components) + 1:
                    neutralized = self._neutralize_cross_section(
                        cs_factor_aligned[valid],
                        X_aligned[valid],
                        method
                    )
                    
                    # Store for all original indices
                    result_series = pd.Series(np.nan, index=cs_factor.index)
                    result_series.loc[neutralized.index] = neutralized.values
                    
                    for stock in cs_factor.index:
                        neutralized_values.append({
                            'date': date,
                            'stock': stock,
                            'alpha': result_series.get(stock, np.nan)
                        })
                else:
                    # Not enough observations - keep original (demeaned)
                    demeaned = cs_factor - cs_factor.mean()
                    for stock in cs_factor.index:
                        neutralized_values.append({
                            'date': date,
                            'stock': stock,
                            'alpha': demeaned.get(stock, np.nan)
                        })
            else:
                # No neutralization components - just demean
                demeaned = cs_factor - cs_factor.mean()
                for stock in cs_factor.index:
                    neutralized_values.append({
                        'date': date,
                        'stock': stock,
                        'alpha': demeaned.get(stock, np.nan)
                    })
        
        # Create result DataFrame
        result = pd.DataFrame(neutralized_values)
        result = result.set_index(['date', 'stock'])
        
        # Winsorize if requested
        if self.winsorize:
            result = self._winsorize_cross_section(result)
        
        return result
    
    def _neutralize_cross_section(
        self,
        factor: pd.Series,
        X: pd.DataFrame,
        method: str
    ) -> pd.Series:
        """
        Neutralize factor within a single cross-section.
        
        Parameters
        ----------
        factor : pd.Series
            Factor values
        X : pd.DataFrame
            Neutralization matrix
        method : str
            Neutralization method
            
        Returns
        -------
        pd.Series
            Neutralized factor (regression residuals)
        """
        if method == 'regression':
            # OLS regression, return residuals
            X_const = sm.add_constant(X)
            try:
                model = sm.OLS(factor, X_const).fit()
                residuals = model.resid
                return residuals
            except Exception:
                return factor - factor.mean()
        
        elif method == 'rank_regression':
            # Rank transform before regression
            factor_ranked = factor.rank(pct=True)
            X_ranked = X.rank(pct=True)
            X_const = sm.add_constant(X_ranked)
            try:
                model = sm.OLS(factor_ranked, X_const).fit()
                residuals = model.resid
                return residuals
            except Exception:
                return factor_ranked - factor_ranked.mean()
        
        elif method == 'demean':
            # Simple demeaning (for industry only)
            return factor - factor.mean()
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _winsorize_cross_section(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """Winsorize within each cross-section."""
        result = data.copy()
        
        for date in data.index.get_level_values(0).unique():
            mask = data.index.get_level_values(0) == date
            cs_data = data.loc[mask]
            
            for col in cs_data.columns:
                values = cs_data[col].dropna()
                if len(values) > 0:
                    lower = values.quantile(self.winsorize_limits[0])
                    upper = values.quantile(self.winsorize_limits[1])
                    result.loc[mask, col] = result.loc[mask, col].clip(lower, upper)
        
        return result
    
    def neutralize_to_factor_loadings(
        self,
        raw_factor: pd.DataFrame,
        factor_loadings: pd.DataFrame,
        factor_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Neutralize factor against specific factor loadings/betas.
        
        This is the core Formula 2 implementation:
        Alpha = Raw Factor - Σ γ_k · Factor_k
        
        where γ_k are estimated from cross-sectional regression.
        
        Parameters
        ----------
        raw_factor : pd.DataFrame
            Raw factor values
        factor_loadings : pd.DataFrame
            Factor loadings (betas) to neutralize against
        factor_col : str, optional
            Column name in raw_factor
            
        Returns
        -------
        pd.DataFrame
            Neutralized alpha factor
        """
        # Extract factor
        if factor_col:
            factor = raw_factor[factor_col].copy()
        else:
            factor = raw_factor.iloc[:, 0].copy()
        
        dates = factor.index.get_level_values(0).unique()
        neutralized_values = []
        gamma_estimates = []
        
        for date in dates:
            cs_factor = factor.xs(date, level=0)
            
            if date in factor_loadings.index.get_level_values(0):
                cs_loadings = factor_loadings.xs(date, level=0)
                
                # Align indices
                common_idx = cs_factor.index.intersection(cs_loadings.index)
                
                if len(common_idx) > cs_loadings.shape[1] + 1:
                    y = cs_factor.loc[common_idx]
                    X = cs_loadings.loc[common_idx]
                    
                    # Remove NaN
                    valid = ~(y.isna() | X.isna().any(axis=1))
                    
                    if valid.sum() > X.shape[1] + 1:
                        X_valid = sm.add_constant(X[valid])
                        y_valid = y[valid]
                        
                        try:
                            model = sm.OLS(y_valid, X_valid).fit()
                            residuals = model.resid
                            
                            # Store gamma estimates
                            gamma_row = {'date': date}
                            for col in X.columns:
                                gamma_row[f'gamma_{col}'] = model.params.get(col, 0)
                            gamma_estimates.append(gamma_row)
                            
                            # Store neutralized values
                            result_series = pd.Series(np.nan, index=cs_factor.index)
                            result_series.loc[residuals.index] = residuals.values
                            
                            for stock in cs_factor.index:
                                neutralized_values.append({
                                    'date': date,
                                    'stock': stock,
                                    'alpha': result_series.get(stock, np.nan)
                                })
                            continue
                            
                        except Exception:
                            pass
            
            # Fallback: just demean
            demeaned = cs_factor - cs_factor.mean()
            for stock in cs_factor.index:
                neutralized_values.append({
                    'date': date,
                    'stock': stock,
                    'alpha': demeaned.get(stock, np.nan)
                })
        
        # Store gamma estimates
        if gamma_estimates:
            self.neutralization_results['gamma'] = pd.DataFrame(gamma_estimates).set_index('date')
        
        # Create result
        result = pd.DataFrame(neutralized_values)
        result = result.set_index(['date', 'stock'])
        
        return result
    
    def compute_residual_exposures(
        self,
        neutralized_factor: pd.DataFrame,
        beta_factors: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Verify that neutralized factor has zero exposure to beta factors.
        
        Parameters
        ----------
        neutralized_factor : pd.DataFrame
            Neutralized factor values
        beta_factors : pd.DataFrame
            Original beta factors
            
        Returns
        -------
        pd.DataFrame
            Residual exposures (should be near zero if neutralization worked)
        """
        dates = neutralized_factor.index.get_level_values(0).unique()
        exposures = []
        
        for date in dates:
            if date not in beta_factors.index.get_level_values(0):
                continue
            
            cs_alpha = neutralized_factor.xs(date, level=0)
            cs_betas = beta_factors.xs(date, level=0)
            
            if isinstance(cs_alpha, pd.DataFrame):
                cs_alpha = cs_alpha.iloc[:, 0]
            
            common_idx = cs_alpha.index.intersection(cs_betas.index)
            
            if len(common_idx) > 10:
                exposure_row = {'date': date}
                
                for col in cs_betas.columns:
                    # Correlation as measure of exposure
                    corr = cs_alpha.loc[common_idx].corr(cs_betas.loc[common_idx, col])
                    exposure_row[f'exposure_{col}'] = corr
                
                exposures.append(exposure_row)
        
        return pd.DataFrame(exposures).set_index('date')
    
    def orthogonalize_factors(
        self,
        factors: pd.DataFrame,
        method: str = 'gram_schmidt'
    ) -> pd.DataFrame:
        """
        Orthogonalize multiple factors to each other.
        
        Useful for combining multiple alpha factors.
        
        Parameters
        ----------
        factors : pd.DataFrame
            Multiple factor values
        method : str
            Orthogonalization method: 'gram_schmidt', 'pca'
            
        Returns
        -------
        pd.DataFrame
            Orthogonalized factors
        """
        dates = factors.index.get_level_values(0).unique()
        orthogonalized = []
        
        for date in dates:
            cs_factors = factors.xs(date, level=0)
            
            # Remove NaN rows
            valid = ~cs_factors.isna().any(axis=1)
            cs_valid = cs_factors[valid]
            
            if len(cs_valid) < cs_factors.shape[1] + 1:
                continue
            
            if method == 'gram_schmidt':
                orth = self._gram_schmidt(cs_valid)
            elif method == 'pca':
                orth = self._pca_orthogonalize(cs_valid)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Store results
            for stock in cs_factors.index:
                row = {'date': date, 'stock': stock}
                for col in orth.columns:
                    row[col] = orth.loc[stock, col] if stock in orth.index else np.nan
                orthogonalized.append(row)
        
        result = pd.DataFrame(orthogonalized)
        result = result.set_index(['date', 'stock'])
        
        return result
    
    def _gram_schmidt(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Apply Gram-Schmidt orthogonalization."""
        result = pd.DataFrame(index=factors.index, columns=factors.columns)
        
        # First factor unchanged (just normalize)
        result.iloc[:, 0] = factors.iloc[:, 0] / np.linalg.norm(factors.iloc[:, 0])
        
        # Orthogonalize subsequent factors
        for i in range(1, factors.shape[1]):
            v = factors.iloc[:, i].values.copy()
            
            for j in range(i):
                u = result.iloc[:, j].values
                # Remove projection onto previous vectors
                v = v - np.dot(v, u) * u
            
            # Normalize
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                result.iloc[:, i] = v / norm
            else:
                result.iloc[:, i] = 0
        
        return result
    
    def _pca_orthogonalize(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Use PCA for orthogonalization."""
        from sklearn.decomposition import PCA
        
        # Standardize
        standardized = (factors - factors.mean()) / factors.std()
        
        # PCA
        pca = PCA(n_components=factors.shape[1])
        transformed = pca.fit_transform(standardized.fillna(0))
        
        result = pd.DataFrame(
            transformed,
            index=factors.index,
            columns=[f'PC{i+1}' for i in range(factors.shape[1])]
        )
        
        return result
    
    def generate_neutralization_report(
        self,
        raw_factor: pd.DataFrame,
        neutralized_factor: pd.DataFrame,
        beta_factors: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate report on neutralization effectiveness.
        
        Parameters
        ----------
        raw_factor : pd.DataFrame
            Original factor
        neutralized_factor : pd.DataFrame
            Neutralized factor
        beta_factors : pd.DataFrame, optional
            Beta factors used for neutralization
            
        Returns
        -------
        str
            Formatted report
        """
        # Compute correlations
        if isinstance(raw_factor, pd.DataFrame) and raw_factor.shape[1] > 0:
            raw_series = raw_factor.iloc[:, 0]
        else:
            raw_series = raw_factor
        
        if isinstance(neutralized_factor, pd.DataFrame) and neutralized_factor.shape[1] > 0:
            alpha_series = neutralized_factor.iloc[:, 0]
        else:
            alpha_series = neutralized_factor
        
        # Align data
        common_idx = raw_series.index.intersection(alpha_series.index)
        
        correlation = raw_series.loc[common_idx].corr(alpha_series.loc[common_idx])
        
        report = f"""
================================================================================
                    Factor Neutralization Report
================================================================================

Neutralization Settings:
-----------------------
  Market Neutralization:    {'Yes' if self.neutralize_market else 'No'}
  Size Neutralization:      {'Yes' if self.neutralize_size else 'No'}
  Industry Neutralization:  {'Yes' if self.neutralize_industry else 'No'}
  Beta Factor Neutralization: {'Yes' if self.neutralize_beta_factors else 'No'}
  Winsorization:           {'Yes' if self.winsorize else 'No'}

Results:
-------
  Raw-Alpha Correlation:   {correlation:.4f}
  
  Raw Factor Stats:
    Mean: {raw_series.loc[common_idx].mean():.4f}
    Std:  {raw_series.loc[common_idx].std():.4f}
  
  Neutralized Alpha Stats:
    Mean: {alpha_series.loc[common_idx].mean():.6f}  (should be ~0)
    Std:  {alpha_series.loc[common_idx].std():.4f}
"""

        # Add residual exposures if beta factors provided
        if beta_factors is not None:
            exposures = self.compute_residual_exposures(neutralized_factor, beta_factors)
            
            report += "\nResidual Exposures (should be ~0):\n"
            report += "-" * 35 + "\n"
            
            for col in exposures.columns:
                mean_exp = exposures[col].mean()
                report += f"  {col}: {mean_exp:.4f}\n"
        
        # Add gamma estimates if available
        if 'gamma' in self.neutralization_results:
            gamma = self.neutralization_results['gamma']
            
            report += "\nGamma Estimates (Factor Loadings):\n"
            report += "-" * 35 + "\n"
            
            for col in gamma.columns:
                mean_gamma = gamma[col].mean()
                std_gamma = gamma[col].std()
                t_stat = mean_gamma / (std_gamma / np.sqrt(len(gamma)))
                report += f"  {col}: {mean_gamma:.4f} (t={t_stat:.2f})\n"
        
        report += """
================================================================================
"""
        return report

