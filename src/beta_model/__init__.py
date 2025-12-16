"""
Beta Model Module
=================

Risk factor construction and beta estimation:
- Fama-French factor construction (MKT, SMB, HML, MOM, RMW, CMA)
- Beta estimation
- Factor portfolio construction
"""

from .risk_factors import RiskFactorModel

__all__ = ['RiskFactorModel']

