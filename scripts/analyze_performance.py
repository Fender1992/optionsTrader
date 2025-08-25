#!/usr/bin/env python
"""
Performance Analysis Script
Analyzes the historical performance of the options trading strategy
with a $10 initial investment over the past year.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
from src.backtest.performance_analyzer import OptionsBacktestAnalyzer, PerformanceMetrics

def run_performance_analysis():
    """Run comprehensive performance analysis with $10 initial investment."""
    
    print("=" * 60)
    print("OPTIONS TRADING STRATEGY - PERFORMANCE ANALYSIS")
    print("Analyzing performance with $10 initial investment")
    print("=" * 60)
    
    # Initialize analyzer with $10
    analyzer = OptionsBacktestAnalyzer(initial_capital=10.0)
    
    # Run simulation for past year
    print("\nRunning backtest simulation for 2024...")
    metrics = analyzer.simulate_year_performance(start_date="2024-01-01")
    
    # Generate report
    report = analyzer.generate_performance_report(metrics)
    print(report)
    
    # Additional analysis
    print("\n" + "=" * 60)
    print("DETAILED ANALYSIS")
    print("=" * 60)
    
    # Investment growth analysis
    print("\nINVESTMENT GROWTH TIMELINE:")
    print("-" * 30)
    
    if metrics.equity_curve:
        milestones = [0, 3, 6, 9, 12]  # Months
        for month in milestones:
            if month == 0:
                value = metrics.equity_curve[0]
                print(f"Start (Jan 2024): ${value:.2f}")
            else:
                # Approximate index for month
                index = min(int(month * 30), len(metrics.equity_curve) - 1)
                value = metrics.equity_curve[index]
                month_name = ["", "Apr", "Jul", "Oct", "Dec"][month // 3]
                print(f"Month {month} ({month_name} 2024): ${value:.2f}")
    
    # Risk/Reward Analysis
    print("\nRISK/REWARD PROFILE:")
    print("-" * 30)
    
    if metrics.avg_win != 0 and metrics.avg_loss != 0:
        risk_reward = abs(metrics.avg_win / metrics.avg_loss)
        print(f"Risk/Reward Ratio: {risk_reward:.2f}:1")
    
    expectancy = (metrics.win_rate * metrics.avg_win) + ((1 - metrics.win_rate) * metrics.avg_loss)
    print(f"Trade Expectancy: ${expectancy:.2f}")
    
    # Best and worst periods
    print("\nBEST PERFORMING PERIODS:")
    print("-" * 30)
    
    if metrics.monthly_returns:
        sorted_months = sorted(metrics.monthly_returns.items(), key=lambda x: x[1], reverse=True)
        for i, (month, return_pct) in enumerate(sorted_months[:3]):
            month_value = 10 * (return_pct / 100)
            print(f"{i+1}. {month}: ${month_value:+.2f} ({return_pct:+.1f}%)")
    
    print("\nWORST PERFORMING PERIODS:")
    print("-" * 30)
    
    if metrics.monthly_returns:
        for i, (month, return_pct) in enumerate(sorted_months[-3:]):
            month_value = 10 * (return_pct / 100)
            print(f"{i+1}. {month}: ${month_value:+.2f} ({return_pct:+.1f}%)")
    
    # Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS & ASSUMPTIONS")
    print("=" * 60)
    
    print("""
ASSUMPTIONS MADE:
-----------------
1. Transaction Costs: $0.65 per contract (industry standard)
2. Slippage: 1% on all options trades
3. Assignment Fee: $15 if assigned on short options
4. Minimum Trade Size: $200 notional (platform requirement)
5. Position Sizing: Max 25% per position, typically 10%
6. Weekly Options Focus: 3-10 day holding periods
7. Strategy Mix: Adaptive based on market conditions
8. Reinvestment: All profits reinvested (compounding)

STRATEGY CHARACTERISTICS:
-------------------------
- Target Delta: 0.35 for directional trades
- Take Profit: 50% gain target
- Stop Loss: 50% loss limit
- IV Rank Threshold: >70% for selling, <30% for buying
- Win Rate Target: 65% through proper selection

IMPORTANT NOTES:
----------------
- Results based on historical market patterns from 2024
- Actual performance will vary based on:
  * Market conditions and volatility
  * Individual stock selection
  * Timing of entry/exit
  * Execution quality
  * Model accuracy

RISK WARNINGS:
--------------
! Options trading involves substantial risk
! 100% loss is possible on any trade
! Past performance doesn't guarantee future results
! This is for educational purposes only
""")
    
    # Create visualization
    create_performance_chart(metrics)
    
    # Save results
    save_results(metrics)
    
    return metrics

def create_performance_chart(metrics: PerformanceMetrics):
    """Create and save performance visualization."""
    
    if not metrics.equity_curve or len(metrics.equity_curve) < 2:
        print("Insufficient data for chart generation")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Equity Curve
    plt.subplot(2, 2, 1)
    plt.plot(metrics.equity_curve, 'b-', linewidth=2)
    plt.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='Initial Investment')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Trade Number')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: Monthly Returns
    if metrics.monthly_returns:
        plt.subplot(2, 2, 2)
        months = list(metrics.monthly_returns.keys())
        returns = list(metrics.monthly_returns.values())
        colors = ['g' if r > 0 else 'r' for r in returns]
        plt.bar(range(len(months)), returns, color=colors)
        plt.title('Monthly Returns (%)')
        plt.xlabel('Month')
        plt.ylabel('Return (%)')
        plt.xticks(range(len(months)), months, rotation=45)
        plt.grid(True, alpha=0.3)
    
    # Subplot 3: Drawdown
    plt.subplot(2, 2, 3)
    peak = metrics.equity_curve[0]
    drawdown = []
    for value in metrics.equity_curve:
        if value > peak:
            peak = value
        dd = ((peak - value) / peak) * 100
        drawdown.append(-dd)
    
    plt.fill_between(range(len(drawdown)), drawdown, 0, color='r', alpha=0.3)
    plt.plot(drawdown, 'r-', linewidth=1)
    plt.title('Drawdown (%)')
    plt.xlabel('Trade Number')
    plt.ylabel('Drawdown (%)')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Win/Loss Distribution
    plt.subplot(2, 2, 4)
    labels = ['Wins', 'Losses']
    sizes = [metrics.winning_trades, metrics.losing_trades]
    colors = ['green', 'red']
    if sum(sizes) > 0:
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        plt.title(f'Win Rate: {metrics.win_rate:.1%}')
    
    plt.tight_layout()
    
    # Save chart
    os.makedirs('artifacts', exist_ok=True)
    plt.savefig('artifacts/performance_analysis.png', dpi=100)
    print("\nPerformance chart saved to artifacts/performance_analysis.png")
    
    plt.show()

def save_results(metrics: PerformanceMetrics):
    """Save analysis results to file."""
    
    results = {
        'analysis_date': datetime.now().isoformat(),
        'initial_investment': 10.0,
        'final_value': 10.0 + metrics.total_return,
        'total_return': metrics.total_return,
        'total_return_percent': metrics.total_return_percent,
        'win_rate': metrics.win_rate,
        'total_trades': metrics.total_trades,
        'sharpe_ratio': metrics.sharpe_ratio,
        'max_drawdown_percent': metrics.max_drawdown_percent,
        'monthly_returns': metrics.monthly_returns,
        'quarterly_returns': metrics.quarterly_returns
    }
    
    os.makedirs('artifacts', exist_ok=True)
    
    # Save as JSON
    with open('artifacts/performance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV for Excel
    df = pd.DataFrame([results])
    df.to_csv('artifacts/performance_summary.csv', index=False)
    
    print("\nResults saved to:")
    print("  - artifacts/performance_results.json")
    print("  - artifacts/performance_summary.csv")

if __name__ == "__main__":
    metrics = run_performance_analysis()
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Initial Investment: $10.00")
    print(f"Final Portfolio Value: ${10 + metrics.total_return:.2f}")
    print(f"Total Return: {metrics.total_return_percent:+.1f}%")
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print("=" * 60)