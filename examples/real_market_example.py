"""
Real Market Factor Mining Example
==================================

使用真实市场数据验证因子挖掘框架的有效性

选择5只代表性股票:
1. AAPL - 苹果 (美股科技龙头)
2. MSFT - 微软 (美股软件巨头)  
3. JPM  - 摩根大通 (美股金融龙头)
4. XOM  - 埃克森美孚 (美股能源龙头)
5. JNJ  - 强生 (美股医疗龙头)

测试周期:
- 日频 (Daily)
- 周频 (Weekly)
- 月频 (Monthly)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.backtest import ICAnalyzer, PortfolioBacktest
from src.beta_model import RiskFactorModel
from src.alpha_model import AlphaFactorNeutralizer
from src.validation import FamaMacBethRegression


# =============================================================================
# 配置
# =============================================================================

# 股票池 - 5只代表性股票
STOCKS = {
    'AAPL': {'name': 'Apple', 'sector': 'Technology'},
    'MSFT': {'name': 'Microsoft', 'sector': 'Technology'},
    'JPM':  {'name': 'JPMorgan', 'sector': 'Financials'},
    'XOM':  {'name': 'ExxonMobil', 'sector': 'Energy'},
    'JNJ':  {'name': 'Johnson & Johnson', 'sector': 'Healthcare'},
}

# 扩展股票池用于更充分的因子测试
EXTENDED_STOCKS = [
    # 科技
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMZN',
    # 金融
    'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC',
    # 能源
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',
    # 医疗
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV',
    # 消费
    'WMT', 'PG', 'KO', 'PEP', 'COST',
    # 工业
    'CAT', 'DE', 'GE', 'HON', 'UPS'
]

# 时间范围
START_DATE = '2020-01-01'
END_DATE = '2024-12-01'


# =============================================================================
# 数据获取函数
# =============================================================================

def download_stock_data(symbols, start_date, end_date, show_progress=True):
    """
    下载股票数据
    
    Parameters
    ----------
    symbols : list
        股票代码列表
    start_date : str
        开始日期
    end_date : str
        结束日期
        
    Returns
    -------
    dict
        包含价格、成交量等数据的字典
    """
    print(f"📊 下载 {len(symbols)} 只股票数据...")
    print(f"   时间范围: {start_date} 至 {end_date}")
    
    # 下载数据
    data = yf.download(
        symbols, 
        start=start_date, 
        end=end_date,
        progress=show_progress,
        auto_adjust=True
    )
    
    # 整理数据
    close = data['Close'] if 'Close' in data.columns else data['Close']
    volume = data['Volume'] if 'Volume' in data.columns else data['Volume']
    high = data['High'] if 'High' in data.columns else data['High']
    low = data['Low'] if 'Low' in data.columns else data['Low']
    
    # 计算收益率
    returns = close.pct_change()
    
    # 计算市值代理 (使用价格 * 成交量作为流动性代理)
    dollar_volume = close * volume
    
    print(f"✓ 下载完成! 共 {len(close)} 个交易日")
    
    return {
        'close': close,
        'returns': returns,
        'volume': volume,
        'high': high,
        'low': low,
        'dollar_volume': dollar_volume
    }


def resample_data(data, freq='W'):
    """
    重采样数据到不同频率
    
    Parameters
    ----------
    data : dict
        原始日频数据
    freq : str
        目标频率: 'D' (日), 'W' (周), 'M' (月)
        
    Returns
    -------
    dict
        重采样后的数据
    """
    if freq == 'D':
        return data
    
    resampled = {}
    
    # 价格取最后值
    resampled['close'] = data['close'].resample(freq).last()
    
    # 成交量取总和
    resampled['volume'] = data['volume'].resample(freq).sum()
    
    # 高低取极值
    resampled['high'] = data['high'].resample(freq).max()
    resampled['low'] = data['low'].resample(freq).min()
    
    # 收益率重新计算
    resampled['returns'] = resampled['close'].pct_change()
    
    # Dollar volume
    resampled['dollar_volume'] = data['dollar_volume'].resample(freq).sum()
    
    return resampled


# =============================================================================
# 因子构建函数
# =============================================================================

def build_momentum_factor(returns, lookback=20, skip=1):
    """
    构建动量因子
    
    Parameters
    ----------
    returns : pd.DataFrame
        收益率数据
    lookback : int
        回溯期
    skip : int
        跳过最近几期
        
    Returns
    -------
    pd.DataFrame
        动量因子值
    """
    # 累计收益（跳过最近skip期）
    momentum = returns.rolling(lookback).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    ).shift(skip)
    
    return momentum


def build_volatility_factor(returns, window=20):
    """
    构建波动率因子（低波动率异象）
    
    Parameters
    ----------
    returns : pd.DataFrame
        收益率数据
    window : int
        计算窗口
        
    Returns
    -------
    pd.DataFrame
        波动率因子（取负使低波动率=高因子值）
    """
    volatility = returns.rolling(window).std()
    # 取负值：低波动率 -> 高因子值
    return -volatility


def build_reversal_factor(returns, window=5):
    """
    构建反转因子
    
    Parameters
    ----------
    returns : pd.DataFrame
        收益率数据
    window : int
        回溯窗口
        
    Returns
    -------
    pd.DataFrame
        反转因子（过去收益取负）
    """
    past_returns = returns.rolling(window).apply(
        lambda x: (1 + x).prod() - 1, raw=True
    )
    # 反转：过去跌的会涨
    return -past_returns


def build_volume_factor(volume, window=20):
    """
    构建成交量异常因子
    
    Parameters
    ----------
    volume : pd.DataFrame
        成交量数据
    window : int
        基准窗口
        
    Returns
    -------
    pd.DataFrame
        成交量异常因子
    """
    avg_volume = volume.rolling(window).mean()
    abnormal_volume = volume / avg_volume - 1
    return abnormal_volume


def build_range_factor(high, low, close, window=20):
    """
    构建价格区间因子（波动性度量）
    
    Parameters
    ----------
    high, low, close : pd.DataFrame
        高低收价格
    window : int
        计算窗口
        
    Returns
    -------
    pd.DataFrame
        价格区间因子
    """
    daily_range = (high - low) / close
    avg_range = daily_range.rolling(window).mean()
    # 取负：低波动 -> 高因子值
    return -avg_range


# =============================================================================
# 因子测试函数
# =============================================================================

def test_factor(factor, returns, factor_name, freq_name, min_periods=10):
    """
    测试单个因子的有效性
    
    Parameters
    ----------
    factor : pd.DataFrame
        因子值
    returns : pd.DataFrame
        收益率
    factor_name : str
        因子名称
    freq_name : str
        频率名称
    min_periods : int
        最小观察期数
        
    Returns
    -------
    dict
        测试结果
    """
    # 转换为长格式
    factor_long = factor.stack()
    factor_long.name = 'factor'
    factor_long.index.names = ['date', 'stock']
    factor_long = factor_long.to_frame()
    
    returns_long = returns.stack()
    returns_long.name = 'return'
    returns_long.index.names = ['date', 'stock']
    returns_long = returns_long.to_frame()
    
    # IC分析
    ic_analyzer = ICAnalyzer(method='spearman', forward_periods=1, min_observations=3)
    
    try:
        ic_series = ic_analyzer.compute_ic(factor_long, returns_long, 
                                           factor_col='factor', return_col='return')
        ic_stats = ic_analyzer.compute_ic_stats()
    except Exception as e:
        return {
            'factor': factor_name,
            'frequency': freq_name,
            'status': 'failed',
            'error': str(e)
        }
    
    # 分组回测（仅当有足够股票时）
    n_stocks = factor.shape[1]
    n_groups = min(5, max(2, n_stocks // 2))
    
    portfolio = PortfolioBacktest(n_groups=n_groups, holding_period=1, long_short=True)
    
    try:
        group_returns = portfolio.construct_portfolios(
            factor_long, returns_long,
            factor_col='factor', return_col='return'
        )
        portfolio_metrics = portfolio.compute_performance_metrics()
        
        ls_return = portfolio_metrics.loc['L-S', 'annual_return'] if 'L-S' in portfolio_metrics.index else np.nan
        ls_sharpe = portfolio_metrics.loc['L-S', 'sharpe_ratio'] if 'L-S' in portfolio_metrics.index else np.nan
    except Exception:
        ls_return = np.nan
        ls_sharpe = np.nan
    
    return {
        'factor': factor_name,
        'frequency': freq_name,
        'ic_mean': ic_stats.get('ic_mean', np.nan),
        'ic_std': ic_stats.get('ic_std', np.nan),
        'ir': ic_stats.get('ir', np.nan),
        't_stat': ic_stats.get('t_stat', np.nan),
        'p_value': ic_stats.get('p_value', np.nan),
        'positive_ratio': ic_stats.get('positive_ratio', np.nan),
        'ls_annual_return': ls_return,
        'ls_sharpe': ls_sharpe,
        'n_periods': ic_stats.get('n_periods', 0),
        'status': 'success'
    }


def run_full_test(data, freq_name):
    """
    运行所有因子测试
    
    Parameters
    ----------
    data : dict
        市场数据
    freq_name : str
        频率名称
        
    Returns
    -------
    list
        测试结果列表
    """
    results = []
    
    # 获取数据
    returns = data['returns'].dropna()
    close = data['close']
    volume = data['volume']
    high = data['high']
    low = data['low']
    
    # 根据频率调整参数
    if freq_name == 'Daily':
        mom_lookback, vol_window, rev_window = 20, 20, 5
    elif freq_name == 'Weekly':
        mom_lookback, vol_window, rev_window = 12, 12, 4
    else:  # Monthly
        mom_lookback, vol_window, rev_window = 6, 6, 3
    
    # 构建并测试各因子
    factors = {
        'Momentum': build_momentum_factor(returns, lookback=mom_lookback),
        'LowVolatility': build_volatility_factor(returns, window=vol_window),
        'Reversal': build_reversal_factor(returns, window=rev_window),
        'Volume': build_volume_factor(volume, window=vol_window),
        'PriceRange': build_range_factor(high, low, close, window=vol_window)
    }
    
    for factor_name, factor_data in factors.items():
        print(f"  测试因子: {factor_name}...")
        result = test_factor(factor_data, returns, factor_name, freq_name)
        results.append(result)
    
    return results


# =============================================================================
# 主程序
# =============================================================================

def main():
    """主程序入口"""
    
    print("="*70)
    print("🔬 Factor Mining Real Market Example")
    print("   真实市场因子挖掘验证")
    print("="*70)
    
    # 下载数据
    print("\n" + "-"*70)
    print("步骤 1: 下载市场数据")
    print("-"*70)
    
    try:
        daily_data = download_stock_data(EXTENDED_STOCKS, START_DATE, END_DATE)
    except Exception as e:
        print(f"❌ 数据下载失败: {e}")
        print("使用简化股票池重试...")
        daily_data = download_stock_data(list(STOCKS.keys()), START_DATE, END_DATE)
    
    # 测试不同频率
    print("\n" + "-"*70)
    print("步骤 2: 多周期因子测试")
    print("-"*70)
    
    all_results = []
    
    for freq, freq_name in [('D', 'Daily'), ('W', 'Weekly'), ('M', 'Monthly')]:
        print(f"\n📅 {freq_name} 频率测试")
        print("-"*40)
        
        # 重采样数据
        freq_data = resample_data(daily_data, freq)
        
        # 运行测试
        results = run_full_test(freq_data, freq_name)
        all_results.extend(results)
    
    # 汇总结果
    print("\n" + "="*70)
    print("📊 测试结果汇总")
    print("="*70)
    
    results_df = pd.DataFrame(all_results)
    
    # 按频率和因子展示
    for freq in ['Daily', 'Weekly', 'Monthly']:
        print(f"\n【{freq} 频率】")
        print("-"*60)
        
        freq_results = results_df[results_df['frequency'] == freq]
        
        for _, row in freq_results.iterrows():
            if row['status'] == 'success':
                significance = "***" if row['p_value'] < 0.01 else \
                              ("**" if row['p_value'] < 0.05 else \
                              ("*" if row['p_value'] < 0.10 else ""))
                
                print(f"  {row['factor']:15s} | "
                      f"IC={row['ic_mean']:+.4f} | "
                      f"IR={row['ir']:.3f} | "
                      f"t={row['t_stat']:+.2f}{significance:3s} | "
                      f"L-S Ret={row['ls_annual_return']*100:+.1f}%")
            else:
                print(f"  {row['factor']:15s} | 测试失败: {row.get('error', 'Unknown')}")
    
    # 找出最有效的因子
    print("\n" + "="*70)
    print("🏆 因子有效性排名 (按 |IR| 排序)")
    print("="*70)
    
    valid_results = results_df[results_df['status'] == 'success'].copy()
    valid_results['abs_ir'] = valid_results['ir'].abs()
    top_results = valid_results.nlargest(10, 'abs_ir')
    
    print("\n排名 | 因子         | 频率    | IC Mean  | IR     | t-stat")
    print("-"*60)
    
    for rank, (_, row) in enumerate(top_results.iterrows(), 1):
        print(f"  {rank:2d}  | {row['factor']:12s} | {row['frequency']:7s} | "
              f"{row['ic_mean']:+.4f} | {row['ir']:.3f} | {row['t_stat']:+.2f}")
    
    # 保存结果
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, 'real_market_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ 结果已保存至: {output_file}")
    
    # 结论
    print("\n" + "="*70)
    print("📝 结论")
    print("="*70)
    
    # 分析哪些因子在哪些频率下有效
    significant_factors = valid_results[valid_results['p_value'] < 0.10]
    
    if len(significant_factors) > 0:
        print("\n✅ 显著有效的因子 (p < 0.10):")
        for _, row in significant_factors.iterrows():
            direction = "正向" if row['ic_mean'] > 0 else "负向"
            print(f"   • {row['factor']} ({row['frequency']}): "
                  f"IC={row['ic_mean']:+.4f}, {direction}预测能力")
    else:
        print("\n⚠️  在当前测试条件下，没有因子达到统计显著水平")
        print("   这可能是因为:")
        print("   1. 样本量较小")
        print("   2. 市场效率较高")
        print("   3. 需要更精细的因子构建方法")
    
    print("\n" + "="*70)
    print("测试完成! 🎉")
    print("="*70)
    
    return results_df


if __name__ == "__main__":
    results = main()

