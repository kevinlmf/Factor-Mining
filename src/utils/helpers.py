"""
Helper Functions
================

Utility functions for data processing and configuration.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Union
import yaml
import os
from datetime import datetime, timedelta


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_sample_data(
    n_stocks: int = 100,
    n_days: int = 1000,
    start_date: str = "2020-01-01",
    seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Create sample data for testing and demonstration.
    
    Parameters
    ----------
    n_stocks : int
        Number of stocks
    n_days : int
        Number of trading days
    start_date : str
        Start date
    seed : int
        Random seed
        
    Returns
    -------
    dict
        Dictionary containing sample data:
        - returns: Stock returns
        - market_cap: Market capitalization
        - book_to_market: Book-to-market ratio
        - momentum: Past returns
        - industry: Industry labels
        - nlp_sentiment: Sample NLP sentiment scores
    """
    np.random.seed(seed)
    
    # Create date index
    dates = pd.date_range(start=start_date, periods=n_days, freq='B')
    
    # Create stock symbols
    stocks = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    # Generate returns with factor structure
    # True factor loadings
    market_beta = np.random.uniform(0.5, 1.5, n_stocks)
    size_loading = np.random.randn(n_stocks)
    value_loading = np.random.randn(n_stocks)
    
    # Factor returns
    market_returns = np.random.randn(n_days) * 0.01
    smb_returns = np.random.randn(n_days) * 0.005
    hml_returns = np.random.randn(n_days) * 0.005
    
    # Stock returns = factor exposures * factor returns + idiosyncratic
    returns = np.zeros((n_days, n_stocks))
    for i in range(n_stocks):
        returns[:, i] = (
            market_beta[i] * market_returns +
            size_loading[i] * smb_returns +
            value_loading[i] * hml_returns +
            np.random.randn(n_days) * 0.02  # Idiosyncratic
        )
    
    returns_df = pd.DataFrame(returns, index=dates, columns=stocks)
    
    # Market cap (log-normal)
    base_cap = np.exp(np.random.randn(n_stocks) * 2 + 10)  # Billions
    market_cap = pd.DataFrame(
        np.outer(np.ones(n_days), base_cap) * 
        np.exp(np.cumsum(returns, axis=0) * 0.5),
        index=dates,
        columns=stocks
    )
    
    # Book-to-market
    base_bm = np.random.uniform(0.3, 2.0, n_stocks)
    book_to_market = pd.DataFrame(
        np.outer(np.ones(n_days), base_bm) * 
        (1 + np.random.randn(n_days, n_stocks) * 0.1),
        index=dates,
        columns=stocks
    )
    
    # Momentum (past 12-month return, skip 1 month)
    momentum = returns_df.rolling(252).apply(
        lambda x: np.prod(1 + x) - 1, raw=True
    ).shift(21)
    
    # Industry (10 industries)
    industries = np.random.choice(
        ['Tech', 'Finance', 'Healthcare', 'Consumer', 'Industrial',
         'Energy', 'Materials', 'Utilities', 'Real Estate', 'Telecom'],
        n_stocks
    )
    industry_df = pd.DataFrame(
        np.tile(industries, (n_days, 1)),
        index=dates,
        columns=stocks
    )
    
    # NLP Sentiment (sample factor with alpha)
    # This has some predictive power for returns
    alpha_signal = np.random.randn(n_stocks)
    nlp_sentiment = pd.DataFrame(
        np.outer(np.ones(n_days), alpha_signal) + 
        np.random.randn(n_days, n_stocks) * 0.5,
        index=dates,
        columns=stocks
    )
    
    # Add some predictability to returns based on sentiment
    future_returns_boost = nlp_sentiment.shift(1).values * 0.001
    returns_df = returns_df + pd.DataFrame(
        np.nan_to_num(future_returns_boost, 0),
        index=dates,
        columns=stocks
    )
    
    return {
        'returns': returns_df,
        'market_cap': market_cap,
        'book_to_market': book_to_market,
        'momentum': momentum,
        'industry': industry_df,
        'nlp_sentiment': nlp_sentiment
    }


def compute_forward_returns(
    returns: pd.DataFrame,
    periods: int = 1,
    cumulative: bool = True
) -> pd.DataFrame:
    """
    Compute forward returns for factor evaluation.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Stock returns (wide format)
    periods : int
        Number of periods forward
    cumulative : bool
        If True, compute cumulative return; else arithmetic sum
        
    Returns
    -------
    pd.DataFrame
        Forward returns
    """
    if cumulative and periods > 1:
        forward = (1 + returns).rolling(periods).apply(
            lambda x: np.prod(x) - 1, raw=True
        ).shift(-periods)
    else:
        forward = returns.shift(-periods)
        if not cumulative and periods > 1:
            forward = returns.rolling(periods).sum().shift(-periods)
    
    return forward


def align_data(
    *dataframes: pd.DataFrame,
    how: str = 'inner'
) -> Tuple[pd.DataFrame, ...]:
    """
    Align multiple DataFrames to common index and columns.
    
    Parameters
    ----------
    *dataframes : pd.DataFrame
        DataFrames to align
    how : str
        Alignment method: 'inner', 'outer', 'left', 'right'
        
    Returns
    -------
    tuple
        Aligned DataFrames
    """
    if not dataframes:
        return tuple()
    
    # Find common index
    if how == 'inner':
        common_index = dataframes[0].index
        common_columns = dataframes[0].columns
        
        for df in dataframes[1:]:
            common_index = common_index.intersection(df.index)
            common_columns = common_columns.intersection(df.columns)
    
    elif how == 'outer':
        common_index = dataframes[0].index
        common_columns = dataframes[0].columns
        
        for df in dataframes[1:]:
            common_index = common_index.union(df.index)
            common_columns = common_columns.union(df.columns)
    
    else:
        raise ValueError(f"Unknown alignment method: {how}")
    
    # Align all DataFrames
    aligned = []
    for df in dataframes:
        aligned.append(df.reindex(index=common_index, columns=common_columns))
    
    return tuple(aligned)


def wide_to_long(
    df: pd.DataFrame,
    value_name: str = 'value'
) -> pd.DataFrame:
    """
    Convert wide format DataFrame to long format with MultiIndex.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide format DataFrame (dates as index, stocks as columns)
    value_name : str
        Name for the value column
        
    Returns
    -------
    pd.DataFrame
        Long format with MultiIndex (date, stock)
    """
    stacked = df.stack()
    stacked.name = value_name
    stacked.index.names = ['date', 'stock']
    return stacked.to_frame()


def long_to_wide(
    df: pd.DataFrame,
    value_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Convert long format DataFrame to wide format.
    
    Parameters
    ----------
    df : pd.DataFrame
        Long format with MultiIndex (date, stock)
    value_col : str, optional
        Column to unstack; uses first column if not specified
        
    Returns
    -------
    pd.DataFrame
        Wide format (dates as index, stocks as columns)
    """
    if value_col is None:
        if isinstance(df, pd.Series):
            series = df
        else:
            series = df.iloc[:, 0]
    else:
        series = df[value_col]
    
    return series.unstack(level=1)


def save_results(
    results: Dict,
    output_dir: str = "results",
    prefix: str = "factor_analysis"
):
    """
    Save analysis results to files.
    
    Parameters
    ----------
    results : dict
        Dictionary of results to save
    output_dir : str
        Output directory
    prefix : str
        File prefix
    """
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, data in results.items():
        if isinstance(data, pd.DataFrame):
            filepath = os.path.join(output_dir, f"{prefix}_{name}_{timestamp}.csv")
            data.to_csv(filepath)
        elif isinstance(data, pd.Series):
            filepath = os.path.join(output_dir, f"{prefix}_{name}_{timestamp}.csv")
            data.to_csv(filepath)
        elif isinstance(data, str):
            filepath = os.path.join(output_dir, f"{prefix}_{name}_{timestamp}.txt")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(data)
        elif isinstance(data, dict):
            filepath = os.path.join(output_dir, f"{prefix}_{name}_{timestamp}.yaml")
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False)


def compute_ic_summary(
    factor: pd.DataFrame,
    returns: pd.DataFrame,
    periods: List[int] = [1, 5, 20]
) -> pd.DataFrame:
    """
    Compute IC summary for multiple forward return periods.
    
    Parameters
    ----------
    factor : pd.DataFrame
        Factor values (wide format)
    returns : pd.DataFrame
        Stock returns (wide format)
    periods : List[int]
        Forward return periods to analyze
        
    Returns
    -------
    pd.DataFrame
        IC summary statistics
    """
    from scipy import stats
    
    results = []
    
    for period in periods:
        forward_ret = compute_forward_returns(returns, period)
        
        # Align data
        factor_aligned, forward_aligned = align_data(factor, forward_ret)
        
        # Compute IC per date
        ic_values = []
        for date in factor_aligned.index:
            f = factor_aligned.loc[date].dropna()
            r = forward_aligned.loc[date].dropna()
            
            common = f.index.intersection(r.index)
            if len(common) > 10:
                ic, _ = stats.spearmanr(f.loc[common], r.loc[common])
                ic_values.append(ic)
        
        if ic_values:
            ic_series = pd.Series(ic_values)
            results.append({
                'period': period,
                'ic_mean': ic_series.mean(),
                'ic_std': ic_series.std(),
                'ir': ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
                't_stat': ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series))),
                'positive_ratio': (ic_series > 0).mean()
            })
    
    return pd.DataFrame(results)

