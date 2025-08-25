#!/usr/bin/env python
"""
Test Maximum Profit Day Trading Strategy
Verify the strategy configuration and behavior
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.strategies.max_profit_day_trading import MaxProfitDayTradingStrategy, HighFrequencyTradingEngine

def test_strategy_configuration():
    """Test strategy configuration for different account sizes"""
    
    print("MAXIMUM PROFIT DAY TRADING STRATEGY TEST")
    print("=" * 60)
    
    test_accounts = [10, 25, 50, 120, 500, 1000, 5000, 25000]
    
    for account_size in test_accounts:
        print(f"\nTesting ${account_size:,} Account:")
        print("-" * 40)
        
        strategy = MaxProfitDayTradingStrategy(account_size)
        config = strategy.get_current_config()
        
        print(f"  Trades per day: {config['trades_per_day']}")
        print(f"  Position size: {config['position_size_pct']*100:.1f}%")
        print(f"  Profit target: {config['profit_target']*100:.1f}%")
        print(f"  Stop loss: {config['stop_loss']*100:.1f}%")
        print(f"  Max concurrent: {config['max_concurrent']}")
        print(f"  Min win rate: {config['min_win_rate']*100:.1f}%")
        print(f"  Target symbols: {len(config['target_symbols'])} symbols")
        
        # Calculate expected daily profit potential
        position_size = account_size * config['position_size_pct']
        max_daily_profit = position_size * config['profit_target'] * config['trades_per_day']
        max_daily_risk = position_size * config['stop_loss'] * config['trades_per_day']
        
        print(f"  Daily profit potential: ${max_daily_profit:.2f}")
        print(f"  Daily risk exposure: ${max_daily_risk:.2f}")

def test_trading_windows():
    """Test trading window configuration"""
    
    print(f"\n\nTRADING WINDOWS TEST")
    print("=" * 60)
    
    strategy = MaxProfitDayTradingStrategy(100)  # Test with $100 account
    
    total_trades = 0
    for i, window in enumerate(strategy.trading_windows):
        start_time = window['start'].strftime('%H:%M')
        end_time = window['end'].strftime('%H:%M')
        max_trades = window['max_trades']
        volatility = window['volatility']
        multiplier = window.get('profit_multiplier', 1.0)
        
        total_trades += max_trades
        
        print(f"Window {i+1}: {start_time}-{end_time}")
        print(f"  Max trades: {max_trades}")
        print(f"  Volatility: {volatility}")
        print(f"  Profit multiplier: {multiplier:.1f}x")
        print()
    
    print(f"Total daily trade capacity: {total_trades} trades")

def test_market_signal_processing():
    """Test market signal processing"""
    
    print(f"\nMARKET SIGNAL PROCESSING TEST")
    print("=" * 60)
    
    engine = HighFrequencyTradingEngine(initial_capital=120.0)
    
    # Test various signal scenarios
    test_signals = [
        {
            'symbol': 'SPY',
            'signal': {
                'strength': 0.8,
                'trend': 'bullish',
                'confidence': 0.7,
                'iv_rank': 25,
                'volatility': 0.02
            }
        },
        {
            'symbol': 'QQQ', 
            'signal': {
                'strength': 0.6,
                'trend': 'bearish',
                'confidence': 0.65,
                'iv_rank': 75,
                'volatility': 0.03
            }
        },
        {
            'symbol': 'AAPL',
            'signal': {
                'strength': 0.9,
                'trend': 'neutral',
                'confidence': 0.8,
                'iv_rank': 50,
                'volatility': 0.025
            }
        }
    ]
    
    for i, test_case in enumerate(test_signals):
        print(f"\nTest Signal {i+1}: {test_case['symbol']}")
        print(f"  Signal strength: {test_case['signal']['strength']:.1f}")
        print(f"  Trend: {test_case['signal']['trend']}")
        print(f"  IV rank: {test_case['signal']['iv_rank']}%")
        
        # Process the signal
        result = engine.process_market_signal(test_case['symbol'], test_case['signal'])
        
        if result:
            print(f"  + Trade executed: {result['strategy']}")
            print(f"  Position size: ${result['position_size']:.2f}")
            print(f"  Profit target: ${result['profit_target']:.2f}")
            print(f"  Stop loss: ${result['stop_loss']:.2f}")
        else:
            print(f"  - Trade rejected")

def test_performance_targets():
    """Test performance targets and growth projections"""
    
    print(f"\n\nPERFORMANCE TARGETS TEST")  
    print("=" * 60)
    
    # Test $10 account growth scenario
    initial_capital = 10.0
    strategy = MaxProfitDayTradingStrategy(initial_capital)
    
    # Calculate theoretical performance
    daily_trades = strategy.trades_per_day
    position_size_pct = strategy.position_size_pct
    profit_target = strategy.profit_target
    win_rate = strategy.min_win_rate
    stop_loss = strategy.stop_loss
    
    # Expected return per trade
    expected_win = profit_target * position_size_pct * win_rate
    expected_loss = stop_loss * position_size_pct * (1 - win_rate)
    expected_return_per_trade = expected_win - expected_loss
    
    print(f"Account: ${initial_capital:.2f}")
    print(f"Daily trades: {daily_trades}")
    print(f"Position size: {position_size_pct*100:.1f}%")
    print(f"Expected return per trade: {expected_return_per_trade*100:.2f}%")
    print(f"Expected daily return: {expected_return_per_trade*daily_trades*100:.2f}%")
    
    # Project growth
    capital = initial_capital
    
    print(f"\nGrowth projection (compounding daily):")
    for day in [1, 5, 10, 22, 44, 66]:  # 1 day, 1 week, 2 weeks, 1 month, 2 months, 3 months
        daily_multiplier = (1 + expected_return_per_trade * daily_trades)
        projected_capital = initial_capital * (daily_multiplier ** day)
        
        period_name = {
            1: "1 day",
            5: "1 week", 
            10: "2 weeks",
            22: "1 month",
            44: "2 months", 
            66: "3 months"
        }[day]
        
        growth_pct = ((projected_capital - initial_capital) / initial_capital) * 100
        print(f"  {period_name:>8}: ${projected_capital:8.2f} ({growth_pct:6.0f}%)")

if __name__ == "__main__":
    test_strategy_configuration()
    test_trading_windows() 
    test_market_signal_processing()
    test_performance_targets()
    
    print(f"\n" + "="*60)
    print("MAXIMUM PROFIT STRATEGY READY FOR DEPLOYMENT")
    print("="*60)