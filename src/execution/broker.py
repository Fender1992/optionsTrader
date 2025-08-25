import os
import json
import requests
import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Optional, Any, Tuple
import logging
import hashlib
import time as time_module
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BrokerInterface(ABC):
    @abstractmethod
    def get_account_info(self) -> Dict:
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        pass
    
    @abstractmethod
    def place_order(self, order: Dict) -> Dict:
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        pass

class SafetyGuard:
    def __init__(self, config: Dict):
        self.config = config
        self.execution_mode = os.getenv('EXECUTION_MODE', 'paper')
        self.kill_switch = os.getenv('KILL_SWITCH', 'false').lower() == 'true'
        self.cash_buffer = config.get('execution', {}).get('cash_buffer', 0.03)
        self.max_positions = config.get('options', {}).get('max_positions', 10)
        self.max_weight_per_name = config.get('options', {}).get('max_weight_per_name', 0.15)
        self.trade_days = config.get('execution', {}).get('trade_days', ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])
        self.trade_window = config.get('execution', {}).get('trade_window', {})
        
        self.daily_loss_limit = -0.02
        self.absolute_drawdown_limit = -0.10
        self.initial_capital = None
        self.daily_starting_capital = None
        self.order_history = []
        
    def can_place_order(self, order: Dict, current_positions: List, account_info: Dict) -> Tuple[bool, str]:
        
        if self.kill_switch:
            return False, "KILL_SWITCH is active - all orders blocked"
        
        if self.execution_mode != 'live' and order.get('live', False):
            return False, f"Live orders blocked - EXECUTION_MODE is {self.execution_mode}"
        
        if not self._is_trading_hours():
            return False, "Outside trading hours"
        
        if not self._check_position_limits(order, current_positions):
            return False, f"Position limit exceeded (max {self.max_positions})"
        
        if not self._check_concentration_limits(order, current_positions, account_info):
            return False, f"Concentration limit exceeded (max {self.max_weight_per_name:.1%} per name)"
        
        if not self._check_cash_buffer(order, account_info):
            return False, f"Insufficient cash after buffer ({self.cash_buffer:.1%})"
        
        if not self._check_loss_limits(account_info):
            return False, "Loss limit exceeded"
        
        if self._is_duplicate_order(order):
            return False, "Duplicate order detected"
        
        return True, "Order approved"
    
    def _is_trading_hours(self) -> bool:
        now = datetime.now()
        current_day = now.strftime('%a')
        
        if current_day not in self.trade_days:
            return False
        
        start_time = datetime.strptime(self.trade_window.get('start', '09:35'), '%H:%M').time()
        end_time = datetime.strptime(self.trade_window.get('end', '15:55'), '%H:%M').time()
        current_time = now.time()
        
        return start_time <= current_time <= end_time
    
    def _check_position_limits(self, order: Dict, current_positions: List) -> bool:
        if order.get('action') == 'sell':
            return True
        
        unique_symbols = set(pos.get('symbol') for pos in current_positions)
        
        if order.get('symbol') not in unique_symbols:
            return len(unique_symbols) < self.max_positions
        
        return True
    
    def _check_concentration_limits(self, order: Dict, current_positions: List, account_info: Dict) -> bool:
        if order.get('action') == 'sell':
            return True
        
        total_value = account_info.get('total_equity', 0)
        if total_value <= 0:
            return False
        
        symbol = order.get('symbol')
        order_value = order.get('quantity', 0) * order.get('limit_price', 0) * 100
        
        symbol_value = sum(
            pos.get('market_value', 0) 
            for pos in current_positions 
            if pos.get('symbol') == symbol
        )
        
        total_symbol_value = symbol_value + order_value
        concentration = total_symbol_value / total_value
        
        return concentration <= self.max_weight_per_name
    
    def _check_cash_buffer(self, order: Dict, account_info: Dict) -> bool:
        if order.get('action') == 'sell':
            return True
        
        available_cash = account_info.get('cash', 0)
        required_buffer = account_info.get('total_equity', 0) * self.cash_buffer
        order_cost = order.get('quantity', 0) * order.get('limit_price', 0) * 100
        
        return (available_cash - order_cost) >= required_buffer
    
    def _check_loss_limits(self, account_info: Dict) -> bool:
        current_equity = account_info.get('total_equity', 0)
        
        if self.initial_capital is None:
            self.initial_capital = current_equity
        
        if self.daily_starting_capital is None or self._is_new_trading_day():
            self.daily_starting_capital = current_equity
        
        daily_return = (current_equity - self.daily_starting_capital) / self.daily_starting_capital
        if daily_return < self.daily_loss_limit:
            logger.warning(f"Daily loss limit hit: {daily_return:.2%}")
            return False
        
        total_return = (current_equity - self.initial_capital) / self.initial_capital
        if total_return < self.absolute_drawdown_limit:
            logger.warning(f"Absolute drawdown limit hit: {total_return:.2%}")
            return False
        
        return True
    
    def _is_duplicate_order(self, order: Dict) -> bool:
        order_hash = hashlib.md5(
            json.dumps(order, sort_keys=True).encode()
        ).hexdigest()
        
        recent_orders = [
            o for o in self.order_history 
            if (datetime.now() - o['timestamp']).total_seconds() < 60
        ]
        
        for recent in recent_orders:
            if recent['hash'] == order_hash:
                return True
        
        self.order_history.append({
            'hash': order_hash,
            'timestamp': datetime.now()
        })
        
        if len(self.order_history) > 1000:
            self.order_history = self.order_history[-500:]
        
        return False
    
    def _is_new_trading_day(self) -> bool:
        if not self.order_history:
            return True
        
        last_order_date = self.order_history[-1]['timestamp'].date()
        return datetime.now().date() > last_order_date

class TradierBroker(BrokerInterface):
    def __init__(self, config: Dict):
        self.config = config
        self.env = os.getenv('TRADIER_ENV', 'paper')
        self.account_id = os.getenv('TRADIER_ACCOUNT_ID')
        self.access_token = os.getenv('TRADIER_ACCESS_TOKEN')
        
        if self.env == 'paper':
            self.base_url = 'https://sandbox.tradier.com/v1'
        else:
            self.base_url = 'https://api.tradier.com/v1'
        
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
        
        self.safety_guard = SafetyGuard(config)
        
    def get_account_info(self) -> Dict:
        url = f"{self.base_url}/accounts/{self.account_id}/balances"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if 'balances' in data:
                balances = data['balances']
                return {
                    'account_id': self.account_id,
                    'cash': float(balances.get('cash', 0)),
                    'total_equity': float(balances.get('total_equity', 0)),
                    'option_buying_power': float(balances.get('option_buying_power', 0)),
                    'pending_orders_count': int(balances.get('pending_orders_count', 0))
                }
            
            return {}
            
        except requests.RequestException as e:
            logger.error(f"Failed to get account info: {e}")
            return {}
    
    def get_positions(self) -> List[Dict]:
        url = f"{self.base_url}/accounts/{self.account_id}/positions"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if 'positions' not in data:
                return []
            
            positions = data['positions']
            if positions == 'null':
                return []
            
            if 'position' not in positions:
                return []
            
            position_list = positions['position']
            if not isinstance(position_list, list):
                position_list = [position_list]
            
            formatted_positions = []
            for pos in position_list:
                formatted_positions.append({
                    'symbol': pos.get('symbol'),
                    'quantity': int(pos.get('quantity', 0)),
                    'cost_basis': float(pos.get('cost_basis', 0)),
                    'market_value': float(pos.get('market_value', 0)),
                    'unrealized_pnl': float(pos.get('unrealized_pnl', 0)),
                    'realized_pnl': float(pos.get('realized_pnl', 0))
                })
            
            return formatted_positions
            
        except requests.RequestException as e:
            logger.error(f"Failed to get positions: {e}")
            return []
    
    def place_order(self, order: Dict) -> Dict:
        current_positions = self.get_positions()
        account_info = self.get_account_info()
        
        can_place, reason = self.safety_guard.can_place_order(order, current_positions, account_info)
        
        if not can_place:
            logger.warning(f"Order blocked: {reason}")
            return {
                'status': 'rejected',
                'reason': reason,
                'order': order
            }
        
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        
        if 'strategy' in order and order['strategy'] in ['bull_call_spread', 'bear_put_spread']:
            return self._place_multi_leg_order(order)
        
        data = {
            'class': 'option',
            'symbol': order.get('symbol'),
            'option_symbol': order.get('contract'),
            'side': 'buy_to_open' if order.get('action') == 'buy' else 'sell_to_close',
            'quantity': order.get('quantity', 1),
            'type': 'limit',
            'price': order.get('limit_price'),
            'duration': 'day'
        }
        
        try:
            logger.info(f"Placing order: {data}")
            response = requests.post(url, headers=self.headers, data=data)
            response.raise_for_status()
            result = response.json()
            
            if 'order' in result:
                order_info = result['order']
                return {
                    'status': 'submitted',
                    'order_id': order_info.get('id'),
                    'symbol': order_info.get('symbol'),
                    'quantity': order_info.get('quantity'),
                    'price': order_info.get('price'),
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'status': 'failed',
                'error': result,
                'order': order
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to place order: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'order': order
            }
    
    def _place_multi_leg_order(self, order: Dict) -> Dict:
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        
        legs = order.get('legs', [])
        if len(legs) != 2:
            return {
                'status': 'rejected',
                'reason': 'Multi-leg orders must have exactly 2 legs'
            }
        
        data = {
            'class': 'multileg',
            'symbol': order.get('symbol'),
            'type': 'market',
            'duration': 'day'
        }
        
        for i, leg in enumerate(legs):
            side = 'buy_to_open' if leg['action'] == 'buy' else 'sell_to_open'
            data[f'option_symbol[{i}]'] = leg['contract']
            data[f'side[{i}]'] = side
            data[f'quantity[{i}]'] = leg['quantity']
        
        try:
            logger.info(f"Placing multi-leg order: {data}")
            response = requests.post(url, headers=self.headers, data=data)
            response.raise_for_status()
            result = response.json()
            
            if 'order' in result:
                return {
                    'status': 'submitted',
                    'order_id': result['order'].get('id'),
                    'strategy': order['strategy'],
                    'timestamp': datetime.now().isoformat()
                }
            
            return {
                'status': 'failed',
                'error': result
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to place multi-leg order: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def cancel_order(self, order_id: str) -> bool:
        url = f"{self.base_url}/accounts/{self.account_id}/orders/{order_id}"
        
        try:
            response = requests.delete(url, headers=self.headers)
            response.raise_for_status()
            return True
            
        except requests.RequestException as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict:
        url = f"{self.base_url}/accounts/{self.account_id}/orders/{order_id}"
        
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
            
            if 'order' in data:
                order = data['order']
                return {
                    'order_id': order.get('id'),
                    'status': order.get('status'),
                    'filled_quantity': order.get('executed_quantity', 0),
                    'remaining_quantity': order.get('remaining_quantity', 0),
                    'avg_fill_price': order.get('avg_fill_price', 0)
                }
            
            return {}
            
        except requests.RequestException as e:
            logger.error(f"Failed to get order status for {order_id}: {e}")
            return {}