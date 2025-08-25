import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class OptionsDataInterface:
    def __init__(self, broker: str = "tradier"):
        self.broker = broker
        if broker == "tradier":
            self.api = TradierOptionsData()
        else:
            raise ValueError(f"Unsupported broker: {broker}")
    
    def get_chain(self, symbol: str, dte_range: Tuple[int, int], trade_date: datetime = None) -> pd.DataFrame:
        return self.api.get_chain(symbol, dte_range, trade_date)
    
    def select_contract_by_delta(self, chain: pd.DataFrame, target_delta: float, right: str = 'call') -> Optional[pd.Series]:
        return self.api.select_contract_by_delta(chain, target_delta, right)
    
    def compute_iv_rank(self, symbol: str, window: int = 252) -> float:
        return self.api.compute_iv_rank(symbol, window)

class TradierOptionsData:
    def __init__(self):
        self.env = os.getenv('TRADIER_ENV', 'paper')
        self.access_token = os.getenv('TRADIER_ACCESS_TOKEN')
        
        if self.env == 'paper':
            self.base_url = 'https://sandbox.tradier.com/v1'
        else:
            self.base_url = 'https://api.tradier.com/v1'
        
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
        
        self.iv_history = {}
    
    def get_chain(self, symbol: str, dte_range: Tuple[int, int], trade_date: datetime = None) -> pd.DataFrame:
        if trade_date is None:
            trade_date = datetime.now()
        
        min_exp = trade_date + timedelta(days=dte_range[0])
        max_exp = trade_date + timedelta(days=dte_range[1])
        
        url = f"{self.base_url}/markets/options/chains"
        params = {
            'symbol': symbol,
            'expiration': f"{min_exp.strftime('%Y-%m-%d')}:{max_exp.strftime('%Y-%m-%d')}",
            'greeks': 'true'
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'options' not in data or 'option' not in data['options']:
                logger.warning(f"No options data available for {symbol}")
                return pd.DataFrame()
            
            options = data['options']['option']
            
            if not isinstance(options, list):
                options = [options]
            
            rows = []
            for opt in options:
                greeks = opt.get('greeks', {})
                rows.append({
                    'symbol': opt.get('symbol'),
                    'underlying': opt.get('underlying'),
                    'strike': float(opt.get('strike', 0)),
                    'expiration_date': pd.to_datetime(opt.get('expiration_date')),
                    'option_type': opt.get('option_type'),
                    'bid': float(opt.get('bid', 0)),
                    'ask': float(opt.get('ask', 0)),
                    'last': float(opt.get('last', 0)),
                    'volume': int(opt.get('volume', 0)),
                    'open_interest': int(opt.get('open_interest', 0)),
                    'implied_volatility': float(opt.get('implied_volatility', 0)),
                    'delta': float(greeks.get('delta', 0)),
                    'gamma': float(greeks.get('gamma', 0)),
                    'theta': float(greeks.get('theta', 0)),
                    'vega': float(greeks.get('vega', 0)),
                    'rho': float(greeks.get('rho', 0))
                })
            
            chain_df = pd.DataFrame(rows)
            
            if not chain_df.empty:
                chain_df['dte'] = (chain_df['expiration_date'] - trade_date).dt.days
                chain_df['mid_price'] = (chain_df['bid'] + chain_df['ask']) / 2
                chain_df['spread'] = chain_df['ask'] - chain_df['bid']
                chain_df['spread_pct'] = chain_df['spread'] / chain_df['mid_price']
                
                chain_df = chain_df[
                    (chain_df['dte'] >= dte_range[0]) & 
                    (chain_df['dte'] <= dte_range[1])
                ]
            
            return chain_df
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch options chain for {symbol}: {e}")
            return pd.DataFrame()
    
    def select_contract_by_delta(self, chain: pd.DataFrame, target_delta: float, right: str = 'call') -> Optional[pd.Series]:
        if chain.empty:
            return None
        
        filtered = chain[chain['option_type'] == right].copy()
        
        if filtered.empty:
            return None
        
        filtered = filtered[filtered['bid'] > 0]
        filtered = filtered[filtered['ask'] > 0]
        filtered = filtered[filtered['open_interest'] >= 10]
        filtered = filtered[filtered['spread_pct'] <= 0.2]
        
        if filtered.empty:
            logger.warning(f"No liquid contracts found for {right}")
            return None
        
        filtered['delta_diff'] = abs(abs(filtered['delta']) - target_delta)
        
        best_contract = filtered.nsmallest(1, 'delta_diff')
        
        if best_contract.empty:
            return None
        
        return best_contract.iloc[0]
    
    def compute_iv_rank(self, symbol: str, window: int = 252) -> float:
        url = f"{self.base_url}/markets/history"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=window)
        
        params = {
            'symbol': symbol,
            'interval': 'daily',
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'history' not in data or 'day' not in data['history']:
                logger.warning(f"No historical data for IV rank calculation for {symbol}")
                return 0.5
            
            days = data['history']['day']
            if not isinstance(days, list):
                days = [days]
            
            current_chain = self.get_chain(symbol, (20, 50))
            if current_chain.empty:
                return 0.5
            
            current_iv = current_chain['implied_volatility'].mean()
            
            if symbol not in self.iv_history:
                self.iv_history[symbol] = []
            
            self.iv_history[symbol].append(current_iv)
            
            if len(self.iv_history[symbol]) > window:
                self.iv_history[symbol] = self.iv_history[symbol][-window:]
            
            iv_values = self.iv_history[symbol]
            
            if len(iv_values) < 20:
                return 0.5
            
            rank = sum(1 for iv in iv_values if iv <= current_iv) / len(iv_values)
            
            return rank
            
        except requests.RequestException as e:
            logger.error(f"Failed to compute IV rank for {symbol}: {e}")
            return 0.5
    
    def get_underlying_price(self, symbol: str) -> float:
        url = f"{self.base_url}/markets/quotes"
        params = {'symbols': symbol}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'quotes' in data and 'quote' in data['quotes']:
                quote = data['quotes']['quote']
                if isinstance(quote, list):
                    quote = quote[0]
                return float(quote.get('last', 0))
            
            return 0
            
        except requests.RequestException as e:
            logger.error(f"Failed to get underlying price for {symbol}: {e}")
            return 0