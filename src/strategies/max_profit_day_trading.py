#!/usr/bin/env python
"""
Maximum Profit Day Trading Strategy
Direct implementation for explosive account growth through high-frequency options trading
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

class MaxProfitDayTradingStrategy:
    """
    Aggressive day trading strategy optimized for maximum profit growth
    Removes all conservative logic and focuses purely on capital multiplication
    """
    
    def __init__(self, account_size: float, config_path: str = None):
        self.account_size = account_size
        self.current_positions = 0
        self.daily_trades_count = 0
        self.last_trade_time = None
        
        # Load configuration
        if config_path is None:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     'config', 'max_profit_config.json')
        
        self.config = self._load_config(config_path)
        
        # Configure based on account size for MAXIMUM GROWTH using config
        tier = self._get_account_tier(account_size)
        tier_config = self.config['account_size_tiers'][tier]
        
        self.trades_per_day = tier_config['optimal_trades_per_day']
        self.position_size_pct = tier_config['position_size_pct'] 
        self.profit_target = tier_config['profit_target_pct']
        self.stop_loss = tier_config['stop_loss_pct']
        self.max_concurrent = tier_config['max_concurrent']
        
        # Set win rate based on aggressiveness
        if account_size < 50:
            self.min_win_rate = 0.55
        elif account_size < 500:
            self.min_win_rate = 0.60
        elif account_size < 5000:
            self.min_win_rate = 0.65
        else:
            self.min_win_rate = 0.70
        
        # Load trading windows from config
        self.trading_windows = self._load_trading_windows()
        
        # Load execution rules
        self.execution_rules = self.config['execution_rules']
        
        # Load target instruments
        instruments_config = self.config['target_instruments']
        self.target_symbols = instruments_config['primary_symbols'] + instruments_config['secondary_symbols']
        
        # Profit-maximizing parameters from config
        iv_config = instruments_config['iv_rank_thresholds']
        self.min_iv_rank = iv_config['buy_premium_below']
        self.max_iv_rank = iv_config['sell_premium_above']
        self.min_dte = instruments_config['min_dte']
        self.max_dte = instruments_config['max_dte']
        self.target_delta_range = tuple(instruments_config['target_delta_range'])
    
    def _load_config(self, config_path: str) -> Dict:
        """Load trading configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found at {config_path}, using defaults")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if config file is not available"""
        return {
            "account_size_tiers": {
                "tiny_account": {
                    "range": [10, 50],
                    "optimal_trades_per_day": 8,
                    "position_size_pct": 0.30,
                    "profit_target_pct": 0.40,
                    "stop_loss_pct": 0.25,
                    "max_concurrent": 3
                },
                "small_account": {
                    "range": [50, 500],
                    "optimal_trades_per_day": 12,
                    "position_size_pct": 0.18,
                    "profit_target_pct": 0.32,
                    "stop_loss_pct": 0.20,
                    "max_concurrent": 5
                },
                "growth_account": {
                    "range": [500, 5000],
                    "optimal_trades_per_day": 20,
                    "position_size_pct": 0.12,
                    "profit_target_pct": 0.25,
                    "stop_loss_pct": 0.15,
                    "max_concurrent": 8
                },
                "large_account": {
                    "range": [5000, 999999],
                    "optimal_trades_per_day": 30,
                    "position_size_pct": 0.08,
                    "profit_target_pct": 0.20,
                    "stop_loss_pct": 0.12,
                    "max_concurrent": 12
                }
            },
            "execution_rules": {
                "order_type": "limit",
                "max_spread_tolerance": 0.015,
                "profit_reinvestment_rate": 1.0
            },
            "target_instruments": {
                "primary_symbols": ["SPY", "QQQ"],
                "secondary_symbols": ["AAPL", "MSFT", "NVDA"],
                "min_dte": 0,
                "max_dte": 7,
                "target_delta_range": [0.25, 0.55],
                "iv_rank_thresholds": {
                    "buy_premium_below": 30,
                    "sell_premium_above": 70
                }
            }
        }
    
    def _get_account_tier(self, account_size: float) -> str:
        """Determine account tier based on size"""
        if account_size < 50:
            return "tiny_account"
        elif account_size < 500:
            return "small_account"
        elif account_size < 5000:
            return "growth_account"
        else:
            return "large_account"
    
    def _load_trading_windows(self) -> List[Dict]:
        """Load trading windows from config"""
        windows_config = self.config.get('trading_windows_detail', {})
        windows = []
        
        for window_name, window_data in windows_config.items():
            start_time_str = window_data['start_time']
            end_time_str = window_data['end_time']
            
            # Parse time strings (HH:MM format)
            start_hour, start_min = map(int, start_time_str.split(':'))
            end_hour, end_min = map(int, end_time_str.split(':'))
            
            windows.append({
                'start': time(start_hour, start_min),
                'end': time(end_hour, end_min),
                'max_trades': window_data['target_trades'],
                'volatility': window_data['volatility_level'],
                'profit_multiplier': window_data.get('profit_multiplier', 1.0)
            })
        
        return windows
        
    def should_trade_now(self) -> bool:
        """Check if we should execute trades right now"""
        current_time = datetime.now().time()
        
        # Check if in trading window
        in_window = False
        current_window = None
        
        for window in self.trading_windows:
            if window['start'] <= current_time <= window['end']:
                in_window = True
                current_window = window
                break
        
        if not in_window:
            return False
            
        # Check daily trade limits
        if self.daily_trades_count >= self.trades_per_day:
            return False
            
        # Check position limits
        if self.current_positions >= self.max_concurrent:
            return False
            
        # Throttle trades (minimum 30 seconds between trades for quality)
        if self.last_trade_time:
            time_since_last = datetime.now() - self.last_trade_time
            if time_since_last < timedelta(seconds=30):
                return False
        
        return True
    
    def get_position_size(self, current_capital: float) -> float:
        """Calculate optimal position size for maximum growth"""
        base_size = current_capital * self.position_size_pct
        
        # Increase aggression in high volatility windows
        current_time = datetime.now().time()
        volatility_multiplier = 1.0
        
        for window in self.trading_windows:
            if window['start'] <= current_time <= window['end']:
                if window['volatility'] == 'extreme':
                    volatility_multiplier = 1.3
                elif window['volatility'] == 'high':
                    volatility_multiplier = 1.15
                break
        
        return base_size * volatility_multiplier
    
    def select_optimal_strategy(self, market_signal: Dict) -> str:
        """Select strategy type for maximum profit potential"""
        
        signal_strength = market_signal.get('strength', 0.5)
        iv_rank = market_signal.get('iv_rank', 50)
        trend = market_signal.get('trend', 'neutral')
        
        # Prioritize high-profit strategies
        if signal_strength > 0.8 and iv_rank < 30:
            # Strong directional signal + low IV = Long options for maximum gamma
            if trend == 'bullish':
                return 'long_call'
            elif trend == 'bearish':
                return 'long_put'
            else:
                return 'long_straddle'
                
        elif signal_strength > 0.6 and iv_rank > 70:
            # Moderate signal + high IV = Sell premium for consistent profits
            if trend == 'bullish':
                return 'short_put'
            elif trend == 'bearish':
                return 'short_call'  
            else:
                return 'iron_condor'
                
        elif signal_strength > 0.5:
            # Moderate signal = Spreads for defined risk/reward
            if trend == 'bullish':
                return 'bull_call_spread'
            elif trend == 'bearish':
                return 'bear_put_spread'
            else:
                return 'iron_butterfly'
        
        else:
            # Weak signal = Scalping strategies
            return 'long_call'  # Default to directional for small accounts
    
    def calculate_profit_targets(self, strategy_type: str, position_size: float) -> Dict:
        """Calculate profit and loss targets for maximum returns"""
        
        if strategy_type in ['long_call', 'long_put']:
            # Long options - target big moves
            profit_target = position_size * self.profit_target
            stop_loss = position_size * self.stop_loss
            
        elif strategy_type in ['short_call', 'short_put']:
            # Short options - collect premium
            profit_target = position_size * (self.profit_target * 0.8)  # Lower target
            stop_loss = position_size * (self.stop_loss * 1.2)  # Higher risk
            
        elif strategy_type in ['bull_call_spread', 'bear_put_spread']:
            # Debit spreads
            profit_target = position_size * self.profit_target
            stop_loss = position_size * self.stop_loss
            
        else:
            # Complex spreads
            profit_target = position_size * (self.profit_target * 0.6)
            stop_loss = position_size * (self.stop_loss * 0.8)
        
        return {
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'profit_target_pct': self.profit_target,
            'stop_loss_pct': self.stop_loss
        }
    
    def execute_trade(self, symbol: str, market_signal: Dict, current_capital: float) -> Dict:
        """Execute trade with maximum profit parameters"""
        
        if not self.should_trade_now():
            return {'status': 'rejected', 'reason': 'outside_trading_window'}
        
        # Calculate position size
        position_size = self.get_position_size(current_capital)
        
        # Select optimal strategy
        strategy_type = self.select_optimal_strategy(market_signal)
        
        # Calculate targets
        targets = self.calculate_profit_targets(strategy_type, position_size)
        
        # Create trade order
        trade_order = {
            'symbol': symbol,
            'strategy': strategy_type,
            'position_size': position_size,
            'position_size_pct': position_size / current_capital,
            'profit_target': targets['profit_target'],
            'stop_loss': targets['stop_loss'],
            'profit_target_pct': targets['profit_target_pct'],
            'stop_loss_pct': targets['stop_loss_pct'],
            'timestamp': datetime.now(),
            'expected_return': self._calculate_expected_return(targets, market_signal),
            'max_risk': targets['stop_loss'],
            'max_reward': targets['profit_target']
        }
        
        # Update counters
        self.daily_trades_count += 1
        self.current_positions += 1
        self.last_trade_time = datetime.now()
        
        logger.info(f"Executing {strategy_type} on {symbol} with ${position_size:.2f} position")
        
        return {'status': 'executed', 'order': trade_order}
    
    def _calculate_expected_return(self, targets: Dict, market_signal: Dict) -> float:
        """Calculate expected return for trade"""
        win_probability = max(self.min_win_rate, market_signal.get('confidence', 0.6))
        
        expected_return = (win_probability * targets['profit_target_pct']) - \
                         ((1 - win_probability) * targets['stop_loss_pct'])
        
        return expected_return
    
    def close_position(self, position_id: str, exit_reason: str = 'target_hit'):
        """Close position and update counters"""
        self.current_positions = max(0, self.current_positions - 1)
        logger.info(f"Closed position {position_id} due to {exit_reason}")
    
    def reset_daily_counters(self):
        """Reset daily counters at market open"""
        self.daily_trades_count = 0
        logger.info("Daily trade counters reset")
    
    def get_current_config(self) -> Dict:
        """Get current strategy configuration"""
        return {
            'account_size': self.account_size,
            'trades_per_day': self.trades_per_day,
            'position_size_pct': self.position_size_pct,
            'profit_target': self.profit_target,
            'stop_loss': self.stop_loss,
            'max_concurrent': self.max_concurrent,
            'min_win_rate': self.min_win_rate,
            'current_positions': self.current_positions,
            'daily_trades_count': self.daily_trades_count,
            'target_symbols': self.target_symbols
        }
    
    def update_account_size(self, new_size: float):
        """Update strategy parameters based on new account size"""
        logger.info(f"Updating account size from ${self.account_size:.2f} to ${new_size:.2f}")
        
        # Reinitialize with new account size for optimal parameters
        old_trades = self.daily_trades_count
        old_positions = self.current_positions
        
        self.__init__(new_size)
        
        # Restore state
        self.daily_trades_count = old_trades
        self.current_positions = old_positions


class HighFrequencyTradingEngine:
    """
    High-frequency trading engine for maximum profit execution
    Manages multiple strategies and rapid trade execution
    """
    
    def __init__(self, initial_capital: float = 10.0):
        self.capital = initial_capital
        self.strategy = MaxProfitDayTradingStrategy(initial_capital)
        self.active_positions = {}
        self.trade_history = []
        
    def update_capital(self, new_capital: float):
        """Update capital and adjust strategy accordingly"""
        self.capital = new_capital
        self.strategy.update_account_size(new_capital)
    
    def process_market_signal(self, symbol: str, signal_data: Dict) -> Optional[Dict]:
        """Process market signal and execute trade if profitable"""
        
        # Only trade if signal is strong enough for profit
        if signal_data.get('strength', 0) < 0.4:
            return None
            
        # Execute trade
        result = self.strategy.execute_trade(symbol, signal_data, self.capital)
        
        if result['status'] == 'executed':
            # Store active position
            position_id = f"{symbol}_{datetime.now().strftime('%H%M%S')}"
            self.active_positions[position_id] = result['order']
            self.trade_history.append(result['order'])
            
            return result['order']
        
        return None
    
    def check_exit_conditions(self):
        """Check all positions for exit conditions"""
        positions_to_close = []
        
        for position_id, position in self.active_positions.items():
            # Check time-based exits (for 0DTE options)
            if self._should_exit_position(position):
                positions_to_close.append(position_id)
        
        # Close positions
        for position_id in positions_to_close:
            self.strategy.close_position(position_id)
            del self.active_positions[position_id]
    
    def _should_exit_position(self, position: Dict) -> bool:
        """Determine if position should be closed"""
        # For maximum profit, we rely on profit targets and stop losses
        # This would integrate with real-time price monitoring
        return False  # Placeholder - would check actual P&L
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics"""
        return {
            'current_capital': self.capital,
            'total_trades': len(self.trade_history),
            'active_positions': len(self.active_positions),
            'daily_trades_remaining': self.strategy.trades_per_day - self.strategy.daily_trades_count,
            'strategy_config': self.strategy.get_current_config()
        }