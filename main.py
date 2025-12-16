"""
Factor Mining Framework - Main Entry Point
==========================================

This script demonstrates the complete factor research pipeline:
1. Factor Mining & Pre-screening (IC Analysis, Group Backtesting)
2. Beta Model & Risk Factor Construction
3. Alpha Factor Construction & Neutralization
4. Fama-MacBeth Hypothesis Testing

Usage:
    python main.py [--config CONFIG_PATH] [--use-sample-data]
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.factors import NLPFactorBuilder, MicrostructureFactorBuilder, FundamentalFactorBuilder
from src.beta_model import RiskFactorModel
from src.alpha_model import AlphaFactorNeutralizer
from src.backtest import ICAnalyzer, PortfolioBacktest
from src.validation import FamaMacBethRegression
from src.utils import load_config, create_sample_data, save_results


def print_banner():
    """Print welcome banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    🔬 FACTOR MINING FRAMEWORK 🔬                              ║
║                                                                              ║
║   A comprehensive tool for factor research with:                             ║
║   • Factor Mining & Pre-screening                                            ║
║   • Beta Modeling & Risk Factor Construction                                 ║
║   • Alpha Factor Neutralization                                              ║
║   • Fama-MacBeth Hypothesis Testing                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_full_pipeline(data: dict, config: dict, verbose: bool = True):
    """
    Run the complete factor research pipeline.
    
    Parameters
    ----------
    data : dict
        Dictionary containing all required data
    config : dict
        Configuration dictionary
    verbose : bool
        Whether to print detailed output
    """
    results = {}
    
    # =========================================================================
    # STAGE 1: Factor Mining & Pre-screening
    # =========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 1: Factor Mining & Pre-screening")
        print("="*80)
    
    # 1.1 Prepare data
    returns = data['returns']
    market_cap = data['market_cap']
    
    # 1.2 Construct sample alpha factor (NLP sentiment in this demo)
    if verbose:
        print("\n[1.1] Constructing sample NLP sentiment factor...")
    
    nlp_factor = data['nlp_sentiment']
    
    # Convert to long format for analysis
    nlp_factor_long = nlp_factor.stack()
    nlp_factor_long.name = 'nlp_sentiment'
    nlp_factor_long.index.names = ['date', 'stock']
    nlp_factor_long = nlp_factor_long.to_frame()
    
    returns_long = returns.stack()
    returns_long.name = 'return'
    returns_long.index.names = ['date', 'stock']
    returns_long = returns_long.to_frame()
    
    # 1.3 IC Analysis
    if verbose:
        print("\n[1.2] Running IC Analysis...")
    
    ic_analyzer = ICAnalyzer(
        method=config.get('ic_analysis', {}).get('method', 'spearman'),
        forward_periods=config.get('ic_analysis', {}).get('lag', 1)
    )
    
    ic_series = ic_analyzer.compute_ic(
        nlp_factor_long, 
        returns_long,
        factor_col='nlp_sentiment',
        return_col='return'
    )
    
    ic_stats = ic_analyzer.compute_ic_stats()
    results['ic_stats'] = pd.DataFrame([ic_stats])
    
    if verbose:
        print(ic_analyzer.generate_report("NLP Sentiment Factor"))
    
    # 1.4 Portfolio Backtest
    if verbose:
        print("\n[1.3] Running Portfolio Backtest...")
    
    portfolio = PortfolioBacktest(
        n_groups=config.get('portfolio', {}).get('n_groups', 5),
        holding_period=1,
        long_short=True
    )
    
    group_returns = portfolio.construct_portfolios(
        nlp_factor_long,
        returns_long,
        factor_col='nlp_sentiment',
        return_col='return'
    )
    
    portfolio_metrics = portfolio.compute_performance_metrics()
    results['portfolio_metrics'] = portfolio_metrics
    
    if verbose:
        print(portfolio.generate_report("NLP Sentiment Factor"))
    
    # Check if factor passes initial screening
    min_ic = config.get('ic_analysis', {}).get('min_ic_mean', 0.02)
    min_ir = config.get('ic_analysis', {}).get('min_ir', 0.3)
    
    passed_screening = ic_analyzer.is_significant(min_ic=min_ic, min_ir=min_ir)
    
    if verbose:
        print(f"\n[1.4] Initial Screening Result: {'PASSED ✓' if passed_screening else 'FAILED ✗'}")
    
    # =========================================================================
    # STAGE 2: Beta Model & Risk Factor Construction
    # =========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 2: Beta Model & Risk Factor Construction")
        print("="*80)
    
    # 2.1 Construct risk factors (Formula 1)
    if verbose:
        print("\n[2.1] Constructing Fama-French risk factors...")
    
    risk_model = RiskFactorModel(
        factors=['MKT', 'SMB', 'HML', 'MOM'],
        lookback_window=config.get('beta_model', {}).get('lookback_window', 252)
    )
    
    factor_returns = risk_model.construct_factors(
        returns=returns,
        market_cap=market_cap,
        book_to_market=data['book_to_market'],
        momentum=data['momentum'],
        rebalance_freq='M'
    )
    
    results['factor_returns'] = factor_returns
    
    if verbose:
        print("\nRisk Factor Statistics:")
        print(risk_model.get_factor_statistics())
        print("\nFactor Correlations:")
        print(risk_model.get_factor_correlations().round(3))
    
    # 2.2 Estimate stock betas
    if verbose:
        print("\n[2.2] Estimating stock betas...")
    
    stock_betas = risk_model.estimate_betas(
        stock_returns=returns,
        method='rolling',
        window=126  # 6 months
    )
    
    results['stock_betas'] = stock_betas
    
    if verbose:
        print(f"\nEstimated betas for {stock_betas.index.get_level_values('stock').nunique()} stocks")
        print(f"Sample beta statistics:\n{stock_betas.describe()}")
    
    # =========================================================================
    # STAGE 3: Alpha Factor Construction & Neutralization
    # =========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 3: Alpha Factor Construction & Neutralization")
        print("="*80)
    
    # 3.1 Neutralize alpha factor (Formula 2)
    if verbose:
        print("\n[3.1] Neutralizing alpha factor against beta factors...")
    
    neutralizer = AlphaFactorNeutralizer(
        neutralize_market=True,
        neutralize_size=True,
        neutralize_industry=True,
        neutralize_beta_factors=True
    )
    
    # Prepare market cap in long format
    market_cap_long = market_cap.stack()
    market_cap_long.name = 'market_cap'
    market_cap_long.index.names = ['date', 'stock']
    market_cap_long = market_cap_long.to_frame()
    
    # Prepare industry in long format
    industry_long = data['industry'].stack()
    industry_long.name = 'industry'
    industry_long.index.names = ['date', 'stock']
    industry_long = industry_long.to_frame()
    
    # Neutralize
    alpha_factor = neutralizer.neutralize(
        raw_factor=nlp_factor_long,
        beta_factors=stock_betas[[col for col in stock_betas.columns if col.startswith('beta_')]],
        market_cap=market_cap_long,
        industry=industry_long,
        factor_col='nlp_sentiment',
        method='regression'
    )
    
    results['alpha_factor'] = alpha_factor
    
    if verbose:
        print(neutralizer.generate_neutralization_report(
            nlp_factor_long, alpha_factor, stock_betas
        ))
    
    # 3.2 Re-compute IC for neutralized factor
    if verbose:
        print("\n[3.2] Re-evaluating neutralized alpha factor...")
    
    alpha_ic_analyzer = ICAnalyzer(method='spearman', forward_periods=1)
    alpha_ic_series = alpha_ic_analyzer.compute_ic(
        alpha_factor,
        returns_long,
        factor_col='alpha',
        return_col='return'
    )
    
    alpha_ic_stats = alpha_ic_analyzer.compute_ic_stats()
    results['alpha_ic_stats'] = pd.DataFrame([alpha_ic_stats])
    
    if verbose:
        print("\nNeutralized Alpha Factor IC Statistics:")
        print(alpha_ic_analyzer.generate_report("Neutralized Alpha Factor"))
    
    # =========================================================================
    # STAGE 4: Fama-MacBeth Hypothesis Testing
    # =========================================================================
    if verbose:
        print("\n" + "="*80)
        print("STAGE 4: Fama-MacBeth Hypothesis Testing")
        print("="*80)
    
    # 4.1 Run Fama-MacBeth regression
    if verbose:
        print("\n[4.1] Running Fama-MacBeth regression...")
    
    fm_regression = FamaMacBethRegression(
        newey_west_lags=config.get('fama_macbeth', {}).get('newey_west_lags', 6),
        min_stocks=config.get('fama_macbeth', {}).get('min_stocks_per_period', 30)
    )
    
    fm_results = fm_regression.run_fama_macbeth(
        returns=returns_long,
        alpha_factor=alpha_factor,
        beta_exposures=stock_betas[[col for col in stock_betas.columns if col.startswith('beta_')]],
        include_intercept=True
    )
    
    results['fama_macbeth'] = fm_results
    
    if verbose:
        print(fm_regression.generate_report("NLP Sentiment Alpha"))
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    if verbose:
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        print("\n📊 Factor Research Pipeline Results:")
        print("-" * 40)
        
        # IC Analysis Summary
        print(f"\n1. IC Analysis (Raw Factor):")
        print(f"   IC Mean: {ic_stats['ic_mean']:.4f}")
        print(f"   IR: {ic_stats['ir']:.4f}")
        print(f"   Passed Screening: {'Yes ✓' if passed_screening else 'No ✗'}")
        
        # Neutralized Factor Summary
        print(f"\n2. IC Analysis (Neutralized Alpha):")
        print(f"   IC Mean: {alpha_ic_stats['ic_mean']:.4f}")
        print(f"   IR: {alpha_ic_stats['ir']:.4f}")
        
        # Fama-MacBeth Summary
        if 'hypothesis_tests' in fm_results:
            print(f"\n3. Fama-MacBeth Test Results:")
            
            if 'alpha_pricing' in fm_results['hypothesis_tests']:
                test = fm_results['hypothesis_tests']['alpha_pricing']
                print(f"   Alpha Pricing Test:")
                print(f"   - λ_alpha: {test['lambda_mean']:.6f}")
                print(f"   - t-stat: {test['t_stat']:.3f}")
                print(f"   - p-value: {test['p_value']:.4f}")
                print(f"   - Conclusion: {test['conclusion']}")
            
            if 'model_specification' in fm_results['hypothesis_tests']:
                test = fm_results['hypothesis_tests']['model_specification']
                print(f"\n   Model Specification Test:")
                print(f"   - λ_0: {test['lambda_mean']:.6f}")
                print(f"   - t-stat: {test['t_stat']:.3f}")
                print(f"   - Conclusion: {test['conclusion']}")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Factor Mining Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--use-sample-data',
        action='store_true',
        help='Use generated sample data for demonstration'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress detailed output'
    )
    
    args = parser.parse_args()
    
    # Print banner
    if not args.quiet:
        print_banner()
    
    # Load configuration
    try:
        config = load_config(args.config)
        if not args.quiet:
            print(f"✓ Loaded configuration from {args.config}")
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        config = {}
    
    # Load or generate data
    if args.use_sample_data:
        if not args.quiet:
            print("\n✓ Generating sample data for demonstration...")
        data = create_sample_data(
            n_stocks=100,
            n_days=500,
            start_date="2021-01-01"
        )
    else:
        print("\nNote: Use --use-sample-data to run with generated sample data")
        print("For real data, implement your own data loading logic.")
        data = create_sample_data(n_stocks=100, n_days=500)
    
    # Run the full pipeline
    try:
        results = run_full_pipeline(
            data=data,
            config=config,
            verbose=not args.quiet
        )
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        save_results(results, args.output_dir, "factor_analysis")
        
        if not args.quiet:
            print(f"\n✓ Results saved to {args.output_dir}/")
            print("\n" + "="*80)
            print("Pipeline completed successfully! 🎉")
            print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during pipeline execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

