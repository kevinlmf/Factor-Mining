"""
Validation Module
=================

Statistical validation and hypothesis testing:
- Fama-MacBeth regression
- Time series and cross-sectional tests
- Multiple testing corrections
"""

from .fama_macbeth import FamaMacBethRegression

__all__ = ['FamaMacBethRegression']

