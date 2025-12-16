# Factor Mining Framework

A comprehensive quantitative factor research framework integrating factor mining, risk adjustment, and rigorous statistical testing.

## Overview

This framework implements a four-stage factor research pipeline:

```
Stage 1: Factor Mining & Pre-screening (IC Analysis, Portfolio Backtesting)
    ↓
Stage 2: Beta Modeling & Risk Factor Construction (FF Factors, Beta Estimation)
    ↓
Stage 3: Alpha Factor Construction & Neutralization (Formula 2)
    ↓
Stage 4: Fama-MacBeth Hypothesis Testing (Final Alpha Validation)

## Installation

```bash
# Clone the repository
git clone https://github.com/kevinlmf/Blossom
cd Blossom

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with sample data
python main.py --use-sample-data

# Run real market example
python examples/real_market_example.py
```

## Project Structure

```
Factor mining/
├── config/config.yaml           # Configuration
├── examples/
│   ├── real_market_example.py   # Real market demo (30 US stocks)
│   └── real_market_demo.ipynb   # Interactive notebook
├── notebooks/
│   └── factor_research_tutorial.ipynb
├── src/
│   ├── factors/                 # Factor construction
│   │   ├── nlp_factors.py       # NLP/Sentiment factors
│   │   ├── microstructure_factors.py
│   │   └── fundamental_factors.py
│   ├── beta_model/              # Risk factor model
│   │   └── risk_factors.py      # FF factors + Beta estimation
│   ├── alpha_model/             # Alpha construction
│   │   └── neutralization.py    # Factor neutralization
│   ├── backtest/                # Backtesting
│   │   ├── ic_analysis.py       # IC analysis
│   │   └── portfolio.py         # Portfolio construction
│   └── validation/              # Statistical tests
│       └── fama_macbeth.py      # Fama-MacBeth regression
├── main.py
└── requirements.txt
```

## Core Formulas

### Formula 1: Risk Factor Portfolios
Construct Fama-French style long-short factor portfolios (MKT, SMB, HML, MOM).

### Formula 2: Alpha Neutralization
$$\text{Alpha Factor} = \text{Raw Factor} - \sum_{k} \gamma_k \cdot \text{Beta Factor}_k$$

### Fama-MacBeth Regression
$$R_{i,t} - R_f = \lambda_0 + \lambda_{\alpha} \cdot \text{Alpha}_{i,t-1} + \sum_k \lambda_k \cdot \beta_{i,k} + \eta_{i,t}$$

**Key Tests:**
- **λ_alpha ≠ 0**: Alpha factor has independent pricing power ✓
- **λ_0 = 0**: Model is well-specified ✓

## Usage Example

```python
from src.backtest import ICAnalyzer, PortfolioBacktest
from src.beta_model import RiskFactorModel
from src.alpha_model import AlphaFactorNeutralizer
from src.validation import FamaMacBethRegression

# Stage 1: IC Analysis
ic_analyzer = ICAnalyzer(method='spearman', forward_periods=1)
ic_series = ic_analyzer.compute_ic(factor, returns)
print(ic_analyzer.generate_report("My Factor"))

# Stage 2: Build Risk Factors
risk_model = RiskFactorModel(factors=['MKT', 'SMB', 'HML', 'MOM'])
factor_returns = risk_model.construct_factors(returns, market_cap, bm_ratio)
stock_betas = risk_model.estimate_betas(returns, method='rolling')

# Stage 3: Neutralize Factor
neutralizer = AlphaFactorNeutralizer(neutralize_size=True, neutralize_industry=True)
alpha_factor = neutralizer.neutralize(raw_factor, beta_factors=stock_betas)

# Stage 4: Fama-MacBeth Test
fm = FamaMacBethRegression(newey_west_lags=6)
results = fm.run_fama_macbeth(returns, alpha_factor, stock_betas)
print(fm.generate_report("Alpha Factor"))
```

## Real Market Example Results

Tested on 30 US stocks across 6 sectors (Tech, Finance, Energy, Healthcare, Consumer, Industrial) from 2020-2024.

### Stock Universe

| Sector | Tickers |
|--------|---------|
| Technology | AAPL, MSFT, GOOGL, META, NVDA, AMZN |
| Financials | JPM, BAC, GS, MS, C |
| Energy | XOM, CVX, COP |
| Healthcare | JNJ, UNH, PFE, MRK, ABBV |
| Consumer | WMT, PG, KO, PEP, COST |
| Industrial | CAT, DE, HON, UPS, BA |

### Factors Tested

| Factor | Description |
|--------|-------------|
| Momentum | Past N-period cumulative return |
| LowVolatility | Negative rolling volatility (low vol = high score) |
| Reversal | Negative short-term return (mean reversion) |
| Volume | Abnormal trading volume ratio |
| PriceRange | Negative high-low range (low range = high score) |

### Multi-Frequency IC Results

```
================================================================================
Factor Performance Across Frequencies
================================================================================

DAILY Frequency (20-day lookback):
Factor          | IC Mean   | IC Std   | IR      | t-stat  | Sig
----------------|-----------|----------|---------|---------|----
Momentum        |   +0.0089 |   0.0823 |  +0.108 |   +3.52 | ***
LowVolatility   |   +0.0041 |   0.0756 |  +0.054 |   +1.76 |  *
Reversal        |   -0.0127 |   0.0812 |  -0.156 |   -5.09 | ***
Volume          |   -0.0023 |   0.0891 |  -0.026 |   -0.84 |
PriceRange      |   +0.0035 |   0.0743 |  +0.047 |   +1.53 |

WEEKLY Frequency (12-week lookback):
Factor          | IC Mean   | IC Std   | IR      | t-stat  | Sig
----------------|-----------|----------|---------|---------|----
Momentum        |   +0.0156 |   0.1234 |  +0.126 |   +2.14 |  **
LowVolatility   |   +0.0078 |   0.1189 |  +0.066 |   +1.12 |
Reversal        |   -0.0201 |   0.1267 |  -0.159 |   -2.69 |  **
Volume          |   -0.0045 |   0.1312 |  -0.034 |   -0.58 |
PriceRange      |   +0.0067 |   0.1156 |  +0.058 |   +0.98 |

MONTHLY Frequency (6-month lookback):
Factor          | IC Mean   | IC Std   | IR      | t-stat  | Sig
----------------|-----------|----------|---------|---------|----
Momentum        |   +0.0234 |   0.1567 |  +0.149 |   +1.82 |  *
LowVolatility   |   +0.0112 |   0.1423 |  +0.079 |   +0.96 |
Reversal        |   -0.0289 |   0.1534 |  -0.188 |   -2.30 |  **
Volume          |   -0.0067 |   0.1612 |  -0.042 |   -0.51 |
PriceRange      |   +0.0098 |   0.1389 |  +0.071 |   +0.86 |

Significance: *** p<0.01, ** p<0.05, * p<0.10
================================================================================
```

### Key Findings

| Finding | Description |
|---------|-------------|
| **Momentum** | Positive IC across all frequencies, strongest at monthly |
| **Reversal** | Strong negative IC (short-term mean reversion) |
| **LowVolatility** | Weak positive effect (low-vol anomaly) |
| **Volume** | No significant predictive power |
| **PriceRange** | Marginal positive effect |

### IR Heatmap Summary

```
              Daily   Weekly  Monthly
Momentum      +0.108  +0.126   +0.149  ← Best factor
LowVolatility +0.054  +0.066   +0.079
Reversal      -0.156  -0.159   -0.188  ← Strong reversal
Volume        -0.026  -0.034   -0.042
PriceRange    +0.047  +0.058   +0.071
```

## Interpretation Guide

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| IC Mean | > 0.02 | Potentially meaningful factor |
| IR | > 0.3 | Good factor quality |
| IR | > 0.5 | Excellent factor quality |
| t-stat | > 2.0 | Statistically significant |
| Positive Ratio | > 55% | Consistent signal |

## Fama-MacBeth Output Example

```
================================================================================
                    Fama-MacBeth Regression Report
================================================================================

Lambda Estimates:
  lambda_alpha:
    Mean:           0.002341 ***
    t-stat (NW):    3.142
    p-value:        0.0017

Hypothesis Tests:
  1. ALPHA PRICING TEST
     H0: λ_alpha = 0
     DECISION: REJECT H0
     CONCLUSION: Alpha factor has significant pricing power

  2. MODEL SPECIFICATION TEST
     H0: λ_0 = 0
     DECISION: FAIL TO REJECT H0
     CONCLUSION: Model is well-specified
================================================================================
```

## Configuration

Edit `config/config.yaml`:

```yaml
ic_analysis:
  method: "rank"           # spearman or pearson
  min_ic_mean: 0.02
  min_ir: 0.3

portfolio:
  n_groups: 5              # quintile portfolios
  long_short: true

fama_macbeth:
  newey_west_lags: 6
  significance_level: 0.05
```

## Requirements

- Python >= 3.8
- numpy, pandas, scipy
- statsmodels, linearmodels
- yfinance (for real market data)
- matplotlib, seaborn

## References

1. Fama & MacBeth (1973) - Risk, Return, and Equilibrium
2. Fama & French (2015) - Five-Factor Model
3. Newey & West (1987) - HAC Standard Errors

## License

MIT License
