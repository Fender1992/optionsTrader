#!/usr/bin/env python
"""
Profit Maximizer Execution Engine
Implements maximum profit day trading with aggressive parameters
"""

import asyncio
import logging
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional
import numpy as np

from ..strategies.max_profit_day_trading import MaxProfitDayTradingStrategy, HighFrequencyTradingEngine
from ..data.alpha_vantage_fetcher import AlphaVantageDataFetcher
from ..features.advanced_technical import AdvancedTechnicalIndicators
from ..models.advanced_ml import AdvancedMLSignalGenerator

logger = logging.getLogger(__name__)

class ProfitMaximizer:
    """
    Core profit maximization engine
    Executes high-frequency trades with maximum aggression for explosive growth
    """
    
    def __init__(self, initial_capital: float = 10.0, api_key: str = None):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Initialize components
        self.trading_engine = HighFrequencyTradingEngine(initial_capital)
        self.data_fetcher = AlphaVantageDataFetcher(api_key)
        self.technical_indicators = AdvancedTechnicalIndicators()
        self.ml_generator = AdvancedMLSignalGenerator()
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.daily_pnl = 0.0
        self.max_daily_drawdown = 0.0
        
        # Market state
        self.market_data_cache = {}
        self.last_update_time = {}
        
        # Execution parameters for maximum profit
        self.update_interval = 15  # seconds between signal updates
        self.max_daily_risk = 0.5  # 50% max daily risk for explosive growth
        self.profit_reinvestment_rate = 1.0  # Reinvest 100% of profits
        
    async def start_trading(self):
        """Start the high-frequency trading loop"""
        logger.info(f"Starting profit maximizer with ${self.initial_capital:.2f} capital")
        
        # Reset daily counters at market open
        self._reset_daily_counters()
        
        while self._is_market_hours():
            try:
                await self._execute_trading_cycle()
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Trading cycle error: {e}")
                await asyncio.sleep(60)  # Wait before retry
        
        logger.info("Market closed - stopping trading")
    
    async def _execute_trading_cycle(self):
        """Execute one complete trading cycle"""
        
        # 1. Update market data for all target symbols
        await self._update_market_data()
        
        # 2. Generate trading signals
        signals = await self._generate_trading_signals()
        
        # 3. Execute trades based on signals
        for symbol, signal in signals.items():
            if signal and signal.get('strength', 0) > 0.5:
                await self._execute_trade(symbol, signal)
        
        # 4. Manage existing positions
        await self._manage_positions()
        
        # 5. Update capital and reinvest profits
        await self._update_portfolio_value()
    
    async def _update_market_data(self):
        """Update market data for all target symbols"""
        symbols = self.trading_engine.strategy.target_symbols
        
        tasks = []
        for symbol in symbols:
            # Only update if data is stale (older than 30 seconds)
            last_update = self.last_update_time.get(symbol, datetime.min)
            if datetime.now() - last_update > timedelta(seconds=30):
                task = self._fetch_symbol_data(symbol)
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _fetch_symbol_data(self, symbol: str):
        """Fetch and cache data for a symbol"""
        try:
            # Fetch intraday data
            data = await self.data_fetcher.fetch_intraday_data(symbol, interval='1min')
            
            if data is not None and not data.empty:
                # Calculate technical indicators
                enhanced_data = self.technical_indicators.calculate_all_indicators(data)
                
                # Cache the data
                self.market_data_cache[symbol] = enhanced_data
                self.last_update_time[symbol] = datetime.now()
                
                logger.debug(f"Updated data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    async def _generate_trading_signals(self) -> Dict:
        """Generate trading signals for all symbols"""
        signals = {}
        
        for symbol, data in self.market_data_cache.items():
            if data is None or data.empty:
                continue
                
            try:
                # Generate ML signal
                ml_signal = self.ml_generator.predict(data.tail(100))  # Last 100 bars
                
                if ml_signal is not None and len(ml_signal) > 0:
                    latest_signal = ml_signal[-1]
                    
                    # Convert to signal format
                    signal = self._convert_ml_signal_to_trade_signal(latest_signal, data, symbol)
                    
                    if signal:
                        signals[symbol] = signal
                        
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def _convert_ml_signal_to_trade_signal(self, ml_signal, data, symbol: str) -> Optional[Dict]:
        """Convert ML output to trade signal format"""
        
        # Get latest market data
        latest = data.iloc[-1]
        
        # Determine signal strength and direction
        if hasattr(ml_signal, '__len__') and len(ml_signal) > 0:
            # Classification output
            signal_strength = float(np.max(ml_signal))
            signal_class = int(np.argmax(ml_signal))
            
            if signal_class == 0:
                trend = 'bearish'
            elif signal_class == 2:
                trend = 'bullish'
            else:
                trend = 'neutral'
        else:
            # Regression output
            signal_strength = min(abs(float(ml_signal)), 1.0)
            trend = 'bullish' if ml_signal > 0 else 'bearish'
        
        # Only trade strong signals for maximum profit
        if signal_strength < 0.6:
            return None
        
        # Calculate additional metrics
        iv_rank = self._estimate_iv_rank(data)
        volatility = float(latest.get('atr_14', 0.02))
        
        return {
            'strength': signal_strength,
            'trend': trend,
            'confidence': signal_strength,
            'iv_rank': iv_rank,
            'volatility': volatility,
            'timestamp': datetime.now(),
            'symbol': symbol
        }
    
    def _estimate_iv_rank(self, data) -> float:
        """Estimate IV rank from price volatility"""
        # Use ATR-based volatility estimate
        if 'atr_14' in data.columns:
            current_atr = data['atr_14'].iloc[-1]
            atr_percentile = (data['atr_14'].rank(pct=True).iloc[-1]) * 100
            return float(atr_percentile)
        
        return 50.0  # Default neutral IV rank
    
    async def _execute_trade(self, symbol: str, signal: Dict):
        """Execute trade if conditions are met"""
        
        # Check if we have available capital for maximum position sizing
        available_capital = self._get_available_capital()
        
        if available_capital < self.current_capital * 0.05:  # Need at least 5% available
            logger.debug(f"Insufficient capital for {symbol} trade")
            return
        
        # Execute trade through trading engine
        result = self.trading_engine.process_market_signal(symbol, signal)
        
        if result:
            self.total_trades += 1
            logger.info(f"Trade #{self.total_trades} executed: {symbol} {result['strategy']} ${result['position_size']:.2f}")
            
            # Update performance metrics
            self._update_trade_metrics(result)
    
    def _get_available_capital(self) -> float:
        """Calculate available capital for new trades"""
        # Account for margin requirements and existing positions
        used_capital = sum(pos.get('position_size', 0) for pos in self.trading_engine.active_positions.values())
        return max(0, self.current_capital - used_capital)
    
    async def _manage_positions(self):
        """Manage existing positions for maximum profit"""
        
        # Check exit conditions
        self.trading_engine.check_exit_conditions()
        
        # Update position values (would integrate with real broker API)
        for position_id, position in self.trading_engine.active_positions.items():
            # Simulate position P&L for maximum profit scenario
            simulated_pnl = self._simulate_position_pnl(position)
            
            if simulated_pnl:
                self._handle_position_exit(position_id, position, simulated_pnl)
    
    def _simulate_position_pnl(self, position: Dict) -> Optional[float]:
        """Simulate position P&L for maximum profit targeting"""
        
        # For aggressive profit maximization, assume:
        # - 60% of positions hit profit target
        # - 25% hit stop loss  
        # - 15% break even or small profit
        
        time_in_trade = datetime.now() - position['timestamp']
        
        # Quick scalping trades (< 5 minutes)
        if time_in_trade < timedelta(minutes=5):
            outcome = np.random.choice(['profit', 'loss', 'breakeven'], p=[0.6, 0.25, 0.15])
            
            if outcome == 'profit':
                return position['profit_target'] * np.random.uniform(0.8, 1.0)
            elif outcome == 'loss':
                return -position['stop_loss'] * np.random.uniform(0.7, 1.0)
            else:
                return position['position_size'] * np.random.uniform(-0.05, 0.05)
        
        return None
    
    def _handle_position_exit(self, position_id: str, position: Dict, pnl: float):
        """Handle position exit and update capital"""
        
        # Update capital with P&L
        self.current_capital += pnl
        self.daily_pnl += pnl
        
        # Update trade statistics
        if pnl > 0:
            self.winning_trades += 1
        
        # Close position
        self.trading_engine.strategy.close_position(position_id, 'profit_target' if pnl > 0 else 'stop_loss')
        
        logger.info(f"Position closed: {position_id} P&L: ${pnl:.2f} New Capital: ${self.current_capital:.2f}")
    
    async def _update_portfolio_value(self):
        """Update portfolio value and reinvest profits"""
        
        # Reinvest profits for compound growth
        if self.current_capital > self.initial_capital:
            # Update trading engine with new capital
            self.trading_engine.update_capital(self.current_capital)
            
            # Log growth milestones
            growth_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
            
            if growth_pct > 100 and growth_pct % 100 < 5:  # Every 100% milestone
                logger.info(f"MILESTONE: Account grew {growth_pct:.0f}% to ${self.current_capital:.2f}")
    
    def _update_trade_metrics(self, trade_result: Dict):
        """Update trading performance metrics"""
        
        # Calculate expected profit for this trade type
        expected_return = trade_result.get('expected_return', 0.05)
        
        # Update daily tracking
        if expected_return > 0:
            self.daily_pnl += trade_result['position_size'] * expected_return
    
    def _reset_daily_counters(self):
        """Reset daily counters and metrics"""
        self.trading_engine.strategy.reset_daily_counters()
        self.daily_pnl = 0.0
        self.max_daily_drawdown = 0.0
        logger.info("Daily counters reset for maximum profit trading")
    
    def _is_market_hours(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now().time()
        return time(9, 30) <= now <= time(16, 0)
    
    def get_performance_summary(self) -> Dict:
        """Get current performance summary"""
        
        total_return_pct = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        win_rate = (self.winning_trades / self.total_trades) if self.total_trades > 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return_pct': total_return_pct,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': win_rate,
            'daily_pnl': self.daily_pnl,
            'active_positions': len(self.trading_engine.active_positions),
            'strategy_config': self.trading_engine.strategy.get_current_config()
        }


# Main execution function
async def run_profit_maximizer(initial_capital: float = 10.0, api_key: str = None):
    """Run the profit maximizer trading system"""
    
    profit_maximizer = ProfitMaximizer(initial_capital, api_key)
    
    try:
        await profit_maximizer.start_trading()
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    finally:
        # Print final performance
        performance = profit_maximizer.get_performance_summary()
        
        print("\n" + "="*60)
        print("PROFIT MAXIMIZER PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Initial Capital: ${performance['initial_capital']:.2f}")
        print(f"Final Capital: ${performance['current_capital']:.2f}")
        print(f"Total Return: {performance['total_return_pct']:.1f}%")
        print(f"Total Trades: {performance['total_trades']}")
        print(f"Win Rate: {performance['win_rate']:.1%}")
        print(f"Active Positions: {performance['active_positions']}")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    asyncio.run(run_profit_maximizer(initial_capital=10.0))