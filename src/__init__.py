"""
Factor Mining Framework
=======================

A comprehensive framework for factor mining, risk modeling, and validation.

Stages:
1. Factor Mining & Pre-screening (IC Analysis, Group Backtesting)
2. Beta Model & Risk Factor Construction
3. Alpha Factor Construction & Neutralization
4. Fama-MacBeth Hypothesis Testing
"""

__version__ = "1.0.0"
__author__ = "Factor Mining Team"

from . import factors
from . import beta_model
from . import alpha_model
from . import backtest
from . import validation
from . import utils

