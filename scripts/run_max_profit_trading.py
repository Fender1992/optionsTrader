#!/usr/bin/env python
"""
Run Maximum Profit Day Trading Strategy
Executes high-frequency options trading for explosive account growth
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution.profit_maximizer import run_profit_maximizer
from src.strategies.max_profit_day_trading import MaxProfitDayTradingStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/max_profit_trading.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def display_strategy_config(account_size: float):
    """Display the maximum profit strategy configuration"""
    
    strategy = MaxProfitDayTradingStrategy(account_size)
    config = strategy.get_current_config()
    
    print("\n" + "="*80)
    print("MAXIMUM PROFIT DAY TRADING STRATEGY")
    print("="*80)
    print(f"Account Size: ${config['account_size']:.2f}")
    print(f"Trades Per Day: {config['trades_per_day']} (MAXIMUM FREQUENCY)")
    print(f"Position Size: {config['position_size_pct']*100:.1f}% per trade (AGGRESSIVE)")
    print(f"Profit Target: {config['profit_target']*100:.1f}% (HIGH RETURNS)")
    print(f"Stop Loss: {config['stop_loss']*100:.1f}% (CONTROLLED RISK)")
    print(f"Max Concurrent: {config['max_concurrent']} positions")
    print(f"Min Win Rate: {config['min_win_rate']*100:.1f}%")
    print()
    print("TARGET SYMBOLS (Most Liquid):")
    for symbol in config['target_symbols']:
        print(f"  - {symbol}")
    print()
    print("TRADING WINDOWS (Peak Opportunities):")
    print("  09:30-10:30: Opening volatility (4 trades)")  
    print("  10:30-11:30: Mid-morning momentum (3 trades)")
    print("  12:00-13:00: Lunch reversal (3 trades)")
    print("  13:30-14:30: Afternoon setup (3 trades)")
    print("  14:30-15:30: Power hour (4 trades)")
    print("  15:30-16:00: Close scalping (2 trades)")
    print()
    print("PROFIT MAXIMIZATION RULES:")
    print("  ✓ 100% profit reinvestment (compound growth)")
    print("  ✓ Scale position size with account growth")
    print("  ✓ Target 30-40% gains per winning trade")
    print("  ✓ Limit losses to 20-25% per trade")
    print("  ✓ Only trade during peak liquidity windows")
    print("  ✓ Focus on 0-7 DTE options for maximum gamma")
    print("  ✓ Use limit orders to minimize slippage")
    print("="*80)

def calculate_growth_projections(initial_capital: float):
    """Calculate potential growth projections"""
    
    strategy = MaxProfitDayTradingStrategy(initial_capital)
    
    # Calculate expected return per trade
    win_rate = strategy.min_win_rate
    avg_win = strategy.profit_target * strategy.position_size_pct
    avg_loss = strategy.stop_loss * strategy.position_size_pct
    expected_return_per_trade = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    # Project growth scenarios
    trades_per_month = strategy.trades_per_day * 22  # 22 trading days per month
    
    print(f"\nGROWTH PROJECTIONS (${initial_capital:.2f} starting capital):")
    print("-" * 60)
    print(f"Expected return per trade: {expected_return_per_trade*100:.2f}%")
    print(f"Trades per month: {trades_per_month}")
    print(f"Monthly compound rate: {expected_return_per_trade*trades_per_month*100:.1f}%")
    print()
    
    capital = initial_capital
    for month in range(1, 13):
        # Compound monthly
        monthly_growth = (1 + expected_return_per_trade) ** trades_per_month
        capital *= monthly_growth
        
        milestone = ""
        if capital > initial_capital * 10:
            milestone = " 🎯 10X MILESTONE!"
        elif capital > initial_capital * 5:
            milestone = " 🚀 5X MILESTONE!"
        elif capital > initial_capital * 2:
            milestone = " ✨ 2X MILESTONE!"
        
        print(f"Month {month:2d}: ${capital:8,.2f} ({((capital-initial_capital)/initial_capital)*100:6.0f}%){milestone}")
    
    print()
    print("CONSERVATIVE SCENARIO (50% of projected):")
    conservative_capital = initial_capital
    conservative_return = expected_return_per_trade * 0.5
    
    for month in [3, 6, 9, 12]:
        monthly_trades = trades_per_month * month
        conservative_capital = initial_capital * ((1 + conservative_return) ** monthly_trades)
        print(f"  Month {month:2d}: ${conservative_capital:,.2f}")
    
    print()
    print("AGGRESSIVE SCENARIO (150% of projected):")
    aggressive_capital = initial_capital  
    aggressive_return = expected_return_per_trade * 1.5
    
    for month in [3, 6, 9, 12]:
        monthly_trades = trades_per_month * month
        aggressive_capital = initial_capital * ((1 + aggressive_return) ** monthly_trades)
        print(f"  Month {month:2d}: ${aggressive_capital:,.2f}")

async def main():
    """Main execution function"""
    
    print("MAXIMUM PROFIT OPTIONS DAY TRADING SYSTEM")
    print("="*80)
    
    # Get account size
    try:
        account_size = float(input("Enter starting capital ($10-$50000): $") or "10")
    except ValueError:
        account_size = 10.0
        print(f"Using default capital: ${account_size:.2f}")
    
    # Display strategy configuration
    display_strategy_config(account_size)
    
    # Show growth projections
    calculate_growth_projections(account_size)
    
    # Confirm execution
    print("\nREADY TO START MAXIMUM PROFIT TRADING!")
    print("\nWARNINGS:")
    print("⚠️  This strategy uses AGGRESSIVE position sizing")
    print("⚠️  High frequency trading may result in substantial losses")
    print("⚠️  Only use money you can afford to lose")
    print("⚠️  Past performance does not guarantee future results")
    
    confirm = input("\nStart trading? (yes/no): ").lower().strip()
    
    if confirm in ['yes', 'y']:
        print("\n🚀 LAUNCHING PROFIT MAXIMIZER...")
        
        # Get API key
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY') or os.getenv('ALPHAVANTAGE_API_KEY')
        if not api_key:
            print("ERROR: Alpha Vantage API key required!")
            print("Set ALPHA_VANTAGE_API_KEY environment variable or check .env file")
            return
        
        # Start trading
        try:
            await run_profit_maximizer(initial_capital=account_size, api_key=api_key)
        except KeyboardInterrupt:
            print("\n\nTrading stopped by user")
        except Exception as e:
            logger.error(f"Trading error: {e}")
            print(f"\nTrading stopped due to error: {e}")
    
    else:
        print("Trading cancelled.")

if __name__ == "__main__":
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Run the maximum profit trading system
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSystem shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}")
        print(f"System error: {e}")