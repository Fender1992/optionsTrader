import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class OptionsPosition:
    symbol: str
    contract: str
    quantity: int
    side: str  # 'buy' or 'sell'
    option_type: str  # 'call' or 'put'
    strike: float
    expiration: datetime
    entry_price: float
    current_price: float
    delta: float
    rationale: str
    take_profit: float
    stop_loss: float
    dte: int

class OptionsStrategyEngine:
    def __init__(self, config: Dict, options_data_interface):
        self.config = config.get('options', {})
        self.options_data = options_data_interface
        
        self.default_dte = self.config.get('default_dte', [30, 45])
        self.target_delta = self.config.get('target_delta', 0.35)
        self.put_spread_delta = self.config.get('put_spread_delta', 0.25)
        self.max_positions = self.config.get('max_positions', 10)
        self.max_weight_per_name = self.config.get('max_weight_per_name', 0.15)
        self.min_notional = self.config.get('min_notional_per_trade', 200)
        self.take_profit_pct = self.config.get('take_profit', 0.5)
        self.stop_loss_pct = self.config.get('stop_loss', -0.5)
        self.iv_rank_window = self.config.get('iv_rank_window', 252)
        
        self.current_positions = []
        
    def generate_signals(self, 
                        ranked_symbols: List[Dict],
                        news_features: pd.DataFrame = None,
                        current_positions: List[OptionsPosition] = None) -> List[Dict]:
        
        if current_positions:
            self.current_positions = current_positions
        
        target_book = []
        
        existing_symbols = [pos.symbol for pos in self.current_positions]
        available_slots = self.max_positions - len(self.current_positions)
        
        if available_slots <= 0:
            logger.info("Maximum positions reached, checking for exits only")
            return self._check_exits()
        
        for rank_data in ranked_symbols[:available_slots * 2]:
            symbol = rank_data['symbol']
            signal_strength = rank_data.get('signal_strength', 0)
            prediction = rank_data.get('prediction', 'neutral')
            
            if symbol in existing_symbols:
                continue
            
            if self._should_skip_due_to_news(symbol, news_features):
                logger.info(f"Skipping {symbol} due to earnings/news window")
                continue
            
            iv_rank = self.options_data.compute_iv_rank(symbol, self.iv_rank_window)
            
            chain = self.options_data.get_chain(symbol, tuple(self.default_dte))
            if chain.empty:
                logger.warning(f"No options chain available for {symbol}")
                continue
            
            if prediction == 'bullish' and signal_strength > 0.6:
                position = self._create_bullish_position(symbol, chain, iv_rank, signal_strength)
            elif prediction == 'bearish' and signal_strength > 0.6:
                position = self._create_bearish_position(symbol, chain, iv_rank, signal_strength)
            else:
                continue
            
            if position:
                target_book.append(position)
                
                if len(target_book) >= available_slots:
                    break
        
        exit_signals = self._check_exits()
        target_book.extend(exit_signals)
        
        return target_book
    
    def _create_bullish_position(self, symbol: str, chain: pd.DataFrame, iv_rank: float, signal_strength: float) -> Optional[Dict]:
        
        if iv_rank > 0.8:
            return self._create_bull_call_spread(symbol, chain, signal_strength)
        else:
            contract = self.options_data.select_contract_by_delta(chain, self.target_delta, 'call')
            
            if contract is None:
                return None
            
            return {
                'symbol': symbol,
                'contract': contract['symbol'],
                'action': 'buy',
                'quantity': 1,
                'option_type': 'call',
                'strike': contract['strike'],
                'expiration': contract['expiration_date'],
                'limit_price': contract['ask'],
                'delta': contract['delta'],
                'iv': contract['implied_volatility'],
                'rationale': f"Bullish signal (strength: {signal_strength:.2f}), IV rank: {iv_rank:.2f}",
                'take_profit': contract['ask'] * (1 + self.take_profit_pct),
                'stop_loss': contract['ask'] * (1 + self.stop_loss_pct),
                'dte': contract['dte']
            }
    
    def _create_bearish_position(self, symbol: str, chain: pd.DataFrame, iv_rank: float, signal_strength: float) -> Optional[Dict]:
        
        if iv_rank > 0.8:
            return self._create_bear_put_spread(symbol, chain, signal_strength)
        else:
            contract = self.options_data.select_contract_by_delta(chain, self.target_delta, 'put')
            
            if contract is None:
                return None
            
            return {
                'symbol': symbol,
                'contract': contract['symbol'],
                'action': 'buy',
                'quantity': 1,
                'option_type': 'put',
                'strike': contract['strike'],
                'expiration': contract['expiration_date'],
                'limit_price': contract['ask'],
                'delta': contract['delta'],
                'iv': contract['implied_volatility'],
                'rationale': f"Bearish signal (strength: {signal_strength:.2f}), IV rank: {iv_rank:.2f}",
                'take_profit': contract['ask'] * (1 + self.take_profit_pct),
                'stop_loss': contract['ask'] * (1 + self.stop_loss_pct),
                'dte': contract['dte']
            }
    
    def _create_bull_call_spread(self, symbol: str, chain: pd.DataFrame, signal_strength: float) -> Optional[Dict]:
        
        long_contract = self.options_data.select_contract_by_delta(chain, self.target_delta, 'call')
        if long_contract is None:
            return None
        
        short_strikes = chain[
            (chain['option_type'] == 'call') &
            (chain['strike'] > long_contract['strike']) &
            (chain['expiration_date'] == long_contract['expiration_date'])
        ]
        
        if short_strikes.empty:
            return None
        
        short_contract = short_strikes.iloc[0]
        
        net_debit = long_contract['ask'] - short_contract['bid']
        max_profit = (short_contract['strike'] - long_contract['strike']) - net_debit
        
        return {
            'symbol': symbol,
            'strategy': 'bull_call_spread',
            'legs': [
                {
                    'contract': long_contract['symbol'],
                    'action': 'buy',
                    'quantity': 1,
                    'limit_price': long_contract['ask']
                },
                {
                    'contract': short_contract['symbol'],
                    'action': 'sell',
                    'quantity': 1,
                    'limit_price': short_contract['bid']
                }
            ],
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': net_debit,
            'rationale': f"Bull call spread due to high IV rank, signal: {signal_strength:.2f}",
            'take_profit': net_debit + (max_profit * 0.5),
            'stop_loss': net_debit * 0.5
        }
    
    def _create_bear_put_spread(self, symbol: str, chain: pd.DataFrame, signal_strength: float) -> Optional[Dict]:
        
        long_contract = self.options_data.select_contract_by_delta(chain, self.target_delta, 'put')
        if long_contract is None:
            return None
        
        short_strikes = chain[
            (chain['option_type'] == 'put') &
            (chain['strike'] < long_contract['strike']) &
            (chain['expiration_date'] == long_contract['expiration_date'])
        ]
        
        if short_strikes.empty:
            return None
        
        short_contract = short_strikes.iloc[0]
        
        net_debit = long_contract['ask'] - short_contract['bid']
        max_profit = (long_contract['strike'] - short_contract['strike']) - net_debit
        
        return {
            'symbol': symbol,
            'strategy': 'bear_put_spread',
            'legs': [
                {
                    'contract': long_contract['symbol'],
                    'action': 'buy',
                    'quantity': 1,
                    'limit_price': long_contract['ask']
                },
                {
                    'contract': short_contract['symbol'],
                    'action': 'sell',
                    'quantity': 1,
                    'limit_price': short_contract['bid']
                }
            ],
            'net_debit': net_debit,
            'max_profit': max_profit,
            'max_loss': net_debit,
            'rationale': f"Bear put spread due to high IV rank, signal: {signal_strength:.2f}",
            'take_profit': net_debit + (max_profit * 0.5),
            'stop_loss': net_debit * 0.5
        }
    
    def _should_skip_due_to_news(self, symbol: str, news_features: pd.DataFrame) -> bool:
        if news_features is None or news_features.empty:
            return False
        
        symbol_news = news_features[news_features['ticker'] == symbol]
        
        if symbol_news.empty:
            return False
        
        latest = symbol_news.iloc[-1]
        
        if latest.get('extreme_negative_news', 0) == 1:
            return True
        
        if latest.get('sent_ticker_mean_1d', 0) < -0.3:
            return True
        
        return False
    
    def _check_exits(self) -> List[Dict]:
        exit_signals = []
        
        for position in self.current_positions:
            current_date = datetime.now()
            
            days_held = (current_date - position.entry_date).days if hasattr(position, 'entry_date') else 0
            dte_remaining = position.dte - days_held
            
            if dte_remaining <= 7:
                exit_signals.append({
                    'symbol': position.symbol,
                    'contract': position.contract,
                    'action': 'sell',
                    'quantity': position.quantity,
                    'rationale': f"Time-based exit: DTE < 7 days",
                    'exit_type': 'time'
                })
                continue
            
            pnl_pct = (position.current_price - position.entry_price) / position.entry_price
            
            if pnl_pct >= self.take_profit_pct:
                exit_signals.append({
                    'symbol': position.symbol,
                    'contract': position.contract,
                    'action': 'sell',
                    'quantity': position.quantity,
                    'rationale': f"Take profit: {pnl_pct:.1%} gain",
                    'exit_type': 'take_profit'
                })
            elif pnl_pct <= self.stop_loss_pct:
                exit_signals.append({
                    'symbol': position.symbol,
                    'contract': position.contract,
                    'action': 'sell',
                    'quantity': position.quantity,
                    'rationale': f"Stop loss: {pnl_pct:.1%} loss",
                    'exit_type': 'stop_loss'
                })
        
        return exit_signals