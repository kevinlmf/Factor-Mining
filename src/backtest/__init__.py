"""
Backtest Module
===============

Contains factor backtesting and analysis tools:
- IC Analysis
- Group/Quantile Portfolio Backtesting
"""

from .ic_analysis import ICAnalyzer
from .portfolio import PortfolioBacktest

__all__ = ['ICAnalyzer', 'PortfolioBacktest']

