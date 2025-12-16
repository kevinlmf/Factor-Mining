"""
Alpha Model Module
==================

Alpha factor construction and neutralization:
- Factor neutralization (market, size, industry, etc.)
- Orthogonalization to risk factors
- Alpha factor combination
"""

from .neutralization import AlphaFactorNeutralizer

__all__ = ['AlphaFactorNeutralizer']

