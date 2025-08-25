import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PortfolioSizer:
    def __init__(self, config: Dict):
        self.config = config
        self.options_config = config.get('options', {})
        
        self.max_positions = self.options_config.get('max_positions', 10)
        self.max_weight_per_name = self.options_config.get('max_weight_per_name', 0.15)
        self.min_notional = self.options_config.get('min_notional_per_trade', 200)
        self.cash_buffer = config.get('execution', {}).get('cash_buffer', 0.03)
        
        self.vol_scale = True
        self.equal_weight = False
        
    def size_positions(self, 
                       capital: float,
                       signals: List[Dict],
                       current_positions: List[Dict] = None,
                       volatility_data: Dict[str, float] = None,
                       account_info: Dict = None) -> List[Dict]:
        
        if not signals:
            logger.info("No signals to size")
            return []
        
        # COMPOUNDING: Use current account equity instead of static capital
        if account_info:
            total_equity = account_info.get('total_equity', capital)
            logger.info(f"Using current equity for sizing: ${total_equity:,.2f} (original capital: ${capital:,.2f})")
        else:
            total_equity = capital
            logger.info(f"No account info provided, using capital: ${capital:,.2f}")
        
        available_capital = total_equity * (1 - self.cash_buffer)
        
        if current_positions:
            current_value = sum(pos.get('market_value', 0) for pos in current_positions)
            # For compounding, we use the remaining buying power, not subtract positions
            available_capital = (total_equity - current_value) * (1 - self.cash_buffer)
            
            current_count = len(set(pos.get('symbol') for pos in current_positions))
            available_slots = self.max_positions - current_count
        else:
            available_slots = self.max_positions
        
        if available_capital <= 0:
            logger.warning("No available capital for new positions")
            return []
        
        if available_slots <= 0:
            logger.warning("No available position slots")
            return []
        
        exit_orders = [s for s in signals if s.get('exit_type')]
        entry_orders = [s for s in signals if not s.get('exit_type')]
        
        sized_orders = []
        
        for exit_order in exit_orders:
            sized_orders.append(exit_order)
        
        if not entry_orders:
            return sized_orders
        
        entry_orders = entry_orders[:available_slots]
        
        if self.equal_weight:
            weights = self._calculate_equal_weights(len(entry_orders))
        else:
            weights = self._calculate_signal_weights(entry_orders, volatility_data)
        
        for order, weight in zip(entry_orders, weights):
            position_capital = available_capital * weight
            
            # Use total_equity for max position constraint, not original capital
            position_capital = min(position_capital, total_equity * self.max_weight_per_name)
            
            if position_capital < self.min_notional:
                logger.info(f"Position size ${position_capital:.2f} below minimum for {order['symbol']}")
                continue
            
            if 'strategy' in order and order['strategy'] in ['bull_call_spread', 'bear_put_spread']:
                contracts = self._size_spread(position_capital, order)
            else:
                contracts = self._size_single_option(position_capital, order)
            
            if contracts > 0:
                order['quantity'] = contracts
                order['position_size'] = position_capital
                order['weight'] = weight
                sized_orders.append(order)
                
                logger.info(f"Sized {order['symbol']}: {contracts} contracts, ${position_capital:.2f} ({weight:.1%} weight)")
        
        return sized_orders
    
    def _calculate_equal_weights(self, num_positions: int) -> List[float]:
        if num_positions == 0:
            return []
        
        return [1.0 / num_positions] * num_positions
    
    def _calculate_signal_weights(self, 
                                 signals: List[Dict],
                                 volatility_data: Dict[str, float] = None) -> List[float]:
        
        raw_weights = []
        
        for signal in signals:
            signal_strength = abs(signal.get('signal_strength', 0.5))
            
            base_weight = signal_strength
            
            if self.vol_scale and volatility_data:
                symbol = signal['symbol']
                vol = volatility_data.get(symbol, 0.2)
                
                target_vol = 0.15
                vol_scalar = min(target_vol / max(vol, 0.05), 2.0)
                
                base_weight *= vol_scalar
            
            if signal.get('iv_rank', 0) > 0.8:
                base_weight *= 0.8
            
            raw_weights.append(base_weight)
        
        total_weight = sum(raw_weights)
        if total_weight > 0:
            normalized = [w / total_weight for w in raw_weights]
        else:
            normalized = self._calculate_equal_weights(len(signals))
        
        capped = []
        for w in normalized:
            capped.append(min(w, self.max_weight_per_name))
        
        final_total = sum(capped)
        if final_total > 0:
            return [w / final_total for w in capped]
        
        return capped
    
    def _size_single_option(self, capital: float, order: Dict) -> int:
        option_price = order.get('limit_price', 0)
        
        if option_price <= 0:
            logger.warning(f"Invalid option price for {order['symbol']}: ${option_price}")
            return 0
        
        contract_cost = option_price * 100
        
        contracts = int(capital / contract_cost)
        
        if contracts * contract_cost < self.min_notional:
            return 0
        
        return contracts
    
    def _size_spread(self, capital: float, order: Dict) -> int:
        net_debit = order.get('net_debit', 0)
        
        if net_debit <= 0:
            logger.warning(f"Invalid net debit for spread on {order['symbol']}: ${net_debit}")
            return 0
        
        spread_cost = net_debit * 100
        
        contracts = int(capital / spread_cost)
        
        if contracts * spread_cost < self.min_notional:
            return 0
        
        return contracts
    
    def calculate_kelly_fraction(self, 
                                win_rate: float,
                                avg_win: float,
                                avg_loss: float,
                                kelly_scalar: float = 0.25) -> float:
        
        if avg_loss == 0:
            return 0
        
        b = avg_win / abs(avg_loss)
        p = win_rate
        q = 1 - p
        
        kelly = (p * b - q) / b
        
        kelly = max(0, min(kelly, 1))
        
        conservative_kelly = kelly * kelly_scalar
        
        return conservative_kelly
    
    def rebalance_weights(self,
                         current_positions: List[Dict],
                         target_positions: List[Dict],
                         total_capital: float) -> Dict[str, Dict]:
        
        rebalance_actions = {
            'exits': [],
            'entries': [],
            'adjustments': []
        }
        
        current_symbols = {pos['symbol'] for pos in current_positions}
        target_symbols = {pos['symbol'] for pos in target_positions}
        
        exits = current_symbols - target_symbols
        for symbol in exits:
            pos = next(p for p in current_positions if p['symbol'] == symbol)
            rebalance_actions['exits'].append({
                'symbol': symbol,
                'action': 'sell',
                'quantity': pos['quantity'],
                'rationale': 'Rebalance exit'
            })
        
        entries = target_symbols - current_symbols
        for symbol in entries:
            pos = next(p for p in target_positions if p['symbol'] == symbol)
            rebalance_actions['entries'].append(pos)
        
        keep_symbols = current_symbols & target_symbols
        for symbol in keep_symbols:
            current = next(p for p in current_positions if p['symbol'] == symbol)
            target = next(p for p in target_positions if p['symbol'] == symbol)
            
            current_weight = current['market_value'] / total_capital
            target_weight = target.get('weight', 0)
            
            weight_diff = abs(current_weight - target_weight)
            
            if weight_diff > 0.02:
                if target_weight > current_weight:
                    add_capital = (target_weight - current_weight) * total_capital
                    rebalance_actions['adjustments'].append({
                        'symbol': symbol,
                        'action': 'add',
                        'capital': add_capital,
                        'rationale': f'Rebalance: increase weight from {current_weight:.1%} to {target_weight:.1%}'
                    })
                else:
                    reduce_pct = 1 - (target_weight / current_weight)
                    reduce_qty = int(current['quantity'] * reduce_pct)
                    if reduce_qty > 0:
                        rebalance_actions['adjustments'].append({
                            'symbol': symbol,
                            'action': 'reduce',
                            'quantity': reduce_qty,
                            'rationale': f'Rebalance: reduce weight from {current_weight:.1%} to {target_weight:.1%}'
                        })
        
        return rebalance_actions