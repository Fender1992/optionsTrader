import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from enum import Enum
import math

logger = logging.getLogger(__name__)

class StrategyType(Enum):
    """Options strategy types for different market conditions."""
    # Directional
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    
    # Neutral/Income
    SHORT_STRADDLE = "short_straddle"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    
    # Volatility
    LONG_STRADDLE = "long_straddle"
    LONG_STRANGLE = "long_strangle"
    CALENDAR_SPREAD = "calendar_spread"
    DIAGONAL_SPREAD = "diagonal_spread"
    
    # Advanced
    RATIO_SPREAD = "ratio_spread"
    JADE_LIZARD = "jade_lizard"
    BROKEN_WING_BUTTERFLY = "broken_wing_butterfly"

@dataclass
class OptionGreeks:
    """Greeks for an option contract."""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    implied_volatility: float
    
@dataclass
class OptionContract:
    """Represents an option contract with all details."""
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    mid: float
    volume: int
    open_interest: int
    greeks: OptionGreeks
    dte: int
    moneyness: float  # strike/spot
    
@dataclass
class OptionsPosition:
    """Complete options position with multiple legs."""
    strategy_type: StrategyType
    legs: List[Dict]  # Each leg: {contract, quantity, action}
    net_debit: float
    net_credit: float
    max_profit: float
    max_loss: float
    breakeven_points: List[float]
    probability_profit: float
    expected_value: float
    portfolio_greeks: Dict[str, float]
    risk_reward_ratio: float
    margin_requirement: float
    
class AdvancedOptionsStrategy:
    """
    Advanced options strategy engine with Greeks-based selection,
    multi-strategy support, and sophisticated risk management.
    """
    
    def __init__(self, config: Dict, options_data_interface):
        self.config = config.get('options', {})
        self.options_data = options_data_interface
        
        # Strategy parameters
        self.iv_rank_threshold_high = 0.7  # Sell premium above this
        self.iv_rank_threshold_low = 0.3   # Buy premium below this
        self.min_open_interest = 50
        self.min_volume = 10
        self.max_spread_width = 0.1  # Max bid-ask spread as % of mid
        
        # Risk parameters
        self.max_portfolio_delta = 100  # Max directional exposure
        self.max_portfolio_gamma = 50
        self.max_portfolio_vega = 500
        self.max_portfolio_theta = -100  # Max daily decay
        
        # Position sizing
        self.kelly_fraction = 0.25  # Conservative Kelly
        self.max_position_size = 0.1  # Max 10% per position
        
    def select_optimal_strategy(self,
                               signal_strength: float,
                               signal_direction: str,
                               underlying_price: float,
                               volatility_regime: str,
                               iv_rank: float,
                               market_phase: str,
                               time_horizon: str = "weekly") -> StrategyType:
        """
        Select optimal strategy based on market conditions and signals.
        
        Args:
            signal_strength: ML model confidence (0-1)
            signal_direction: 'bullish', 'bearish', or 'neutral'
            volatility_regime: 'low', 'normal', 'high'
            iv_rank: Current IV percentile (0-1)
            market_phase: 'trending', 'ranging', 'breakout'
            time_horizon: 'weekly' or 'monthly'
        """
        
        # High IV environment - prefer selling strategies
        if iv_rank > self.iv_rank_threshold_high:
            if signal_direction == "bullish" and signal_strength > 0.7:
                return StrategyType.BULL_CALL_SPREAD if time_horizon == "weekly" else StrategyType.DIAGONAL_SPREAD
            elif signal_direction == "bearish" and signal_strength > 0.7:
                return StrategyType.BEAR_PUT_SPREAD
            elif signal_direction == "neutral" or signal_strength < 0.5:
                if market_phase == "ranging":
                    return StrategyType.IRON_CONDOR if volatility_regime != "high" else StrategyType.IRON_BUTTERFLY
                else:
                    return StrategyType.SHORT_STRANGLE
        
        # Low IV environment - prefer buying strategies
        elif iv_rank < self.iv_rank_threshold_low:
            if signal_direction == "bullish" and signal_strength > 0.6:
                return StrategyType.LONG_CALL
            elif signal_direction == "bearish" and signal_strength > 0.6:
                return StrategyType.LONG_PUT
            elif volatility_regime == "low" and market_phase == "breakout":
                return StrategyType.LONG_STRADDLE
        
        # Normal IV environment - mixed strategies
        else:
            if signal_strength > 0.8:
                if signal_direction == "bullish":
                    return StrategyType.BULL_CALL_SPREAD
                elif signal_direction == "bearish":
                    return StrategyType.BEAR_PUT_SPREAD
            elif signal_strength > 0.6:
                if time_horizon == "monthly":
                    return StrategyType.CALENDAR_SPREAD
                else:
                    return StrategyType.RATIO_SPREAD if signal_direction != "neutral" else StrategyType.IRON_CONDOR
            else:
                return StrategyType.IRON_CONDOR
    
    def build_position(self,
                      strategy_type: StrategyType,
                      underlying: str,
                      chain: pd.DataFrame,
                      underlying_price: float,
                      target_dte: Tuple[int, int],
                      account_size: float) -> Optional[OptionsPosition]:
        """
        Build a complete options position with the selected strategy.
        """
        
        # Filter chain for liquidity and DTE
        filtered_chain = self._filter_chain(chain, target_dte)
        
        if filtered_chain.empty:
            logger.warning(f"No liquid contracts found for {underlying}")
            return None
        
        # Build position based on strategy type
        if strategy_type == StrategyType.LONG_CALL:
            return self._build_long_call(filtered_chain, underlying_price, account_size)
        elif strategy_type == StrategyType.BULL_CALL_SPREAD:
            return self._build_bull_call_spread(filtered_chain, underlying_price, account_size)
        elif strategy_type == StrategyType.IRON_CONDOR:
            return self._build_iron_condor(filtered_chain, underlying_price, account_size)
        elif strategy_type == StrategyType.SHORT_STRANGLE:
            return self._build_short_strangle(filtered_chain, underlying_price, account_size)
        # Add more strategies as needed
        else:
            logger.warning(f"Strategy {strategy_type} not yet implemented")
            return None
    
    def _filter_chain(self, chain: pd.DataFrame, target_dte: Tuple[int, int]) -> pd.DataFrame:
        """Filter option chain for liquidity and DTE requirements."""
        
        filtered = chain[
            (chain['dte'] >= target_dte[0]) &
            (chain['dte'] <= target_dte[1]) &
            (chain['open_interest'] >= self.min_open_interest) &
            (chain['volume'] >= self.min_volume) &
            ((chain['ask'] - chain['bid']) / chain['mid_price'] <= self.max_spread_width)
        ].copy()
        
        return filtered
    
    def _build_long_call(self, chain: pd.DataFrame, spot: float, account_size: float) -> OptionsPosition:
        """Build a long call position."""
        
        calls = chain[chain['option_type'] == 'call'].copy()
        
        # Select strike near 0.4 delta for balanced risk/reward
        target_delta = 0.4
        calls['delta_diff'] = abs(calls['delta'] - target_delta)
        best_call = calls.nsmallest(1, 'delta_diff').iloc[0]
        
        # Position sizing
        max_risk = account_size * self.max_position_size
        contracts = int(max_risk / (best_call['ask'] * 100))
        contracts = max(1, contracts)
        
        cost = contracts * best_call['ask'] * 100
        
        return OptionsPosition(
            strategy_type=StrategyType.LONG_CALL,
            legs=[{
                'contract': best_call['symbol'],
                'strike': best_call['strike'],
                'expiration': best_call['expiration_date'],
                'type': 'call',
                'action': 'buy',
                'quantity': contracts,
                'price': best_call['ask']
            }],
            net_debit=cost,
            net_credit=0,
            max_profit=float('inf'),  # Unlimited upside
            max_loss=cost,
            breakeven_points=[best_call['strike'] + best_call['ask']],
            probability_profit=self._calculate_probability_profit(best_call, spot, 'call'),
            expected_value=self._calculate_expected_value(best_call, spot, cost),
            portfolio_greeks={
                'delta': contracts * best_call['delta'] * 100,
                'gamma': contracts * best_call['gamma'] * 100,
                'theta': contracts * best_call['theta'] * 100,
                'vega': contracts * best_call['vega'] * 100
            },
            risk_reward_ratio=3.0,  # Typical for long options
            margin_requirement=0  # Long options don't require margin
        )
    
    def _build_bull_call_spread(self, chain: pd.DataFrame, spot: float, account_size: float) -> OptionsPosition:
        """Build a bull call spread."""
        
        calls = chain[chain['option_type'] == 'call'].copy()
        calls = calls.sort_values('strike')
        
        # Buy ATM/slightly ITM, sell OTM
        long_strike = calls[calls['strike'] <= spot].iloc[-1] if len(calls[calls['strike'] <= spot]) > 0 else calls.iloc[0]
        short_strikes = calls[calls['strike'] > long_strike['strike']]
        
        if short_strikes.empty:
            return None
        
        # Select short strike for good risk/reward (typically 5-10% OTM)
        target_short = spot * 1.05
        short_strikes['diff'] = abs(short_strikes['strike'] - target_short)
        short_strike = short_strikes.nsmallest(1, 'diff').iloc[0]
        
        # Calculate spread metrics
        net_debit = long_strike['ask'] - short_strike['bid']
        max_profit = short_strike['strike'] - long_strike['strike'] - net_debit
        max_loss = net_debit
        
        # Position sizing
        max_risk = account_size * self.max_position_size
        contracts = int(max_risk / (net_debit * 100))
        contracts = max(1, contracts)
        
        return OptionsPosition(
            strategy_type=StrategyType.BULL_CALL_SPREAD,
            legs=[
                {
                    'contract': long_strike['symbol'],
                    'strike': long_strike['strike'],
                    'expiration': long_strike['expiration_date'],
                    'type': 'call',
                    'action': 'buy',
                    'quantity': contracts,
                    'price': long_strike['ask']
                },
                {
                    'contract': short_strike['symbol'],
                    'strike': short_strike['strike'],
                    'expiration': short_strike['expiration_date'],
                    'type': 'call',
                    'action': 'sell',
                    'quantity': contracts,
                    'price': short_strike['bid']
                }
            ],
            net_debit=net_debit * contracts * 100,
            net_credit=0,
            max_profit=max_profit * contracts * 100,
            max_loss=max_loss * contracts * 100,
            breakeven_points=[long_strike['strike'] + net_debit],
            probability_profit=self._calculate_spread_probability(long_strike, short_strike, spot),
            expected_value=self._calculate_spread_expected_value(long_strike, short_strike, spot, net_debit),
            portfolio_greeks=self._calculate_spread_greeks(long_strike, short_strike, contracts),
            risk_reward_ratio=max_profit / max_loss if max_loss > 0 else 0,
            margin_requirement=0  # Debit spreads don't require margin beyond cost
        )
    
    def _build_iron_condor(self, chain: pd.DataFrame, spot: float, account_size: float) -> OptionsPosition:
        """Build an iron condor for neutral markets."""
        
        calls = chain[chain['option_type'] == 'call'].sort_values('strike')
        puts = chain[chain['option_type'] == 'put'].sort_values('strike')
        
        # Select strikes: sell ~0.20 delta, buy further OTM for protection
        target_delta = 0.20
        
        # Short call
        calls['delta_diff'] = abs(abs(calls['delta']) - target_delta)
        short_call = calls.nsmallest(1, 'delta_diff').iloc[0]
        
        # Long call (further OTM)
        long_calls = calls[calls['strike'] > short_call['strike']]
        if long_calls.empty:
            return None
        long_call = long_calls.iloc[0]
        
        # Short put
        puts['delta_diff'] = abs(abs(puts['delta']) - target_delta)
        short_put = puts.nsmallest(1, 'delta_diff').iloc[0]
        
        # Long put (further OTM)
        long_puts = puts[puts['strike'] < short_put['strike']]
        if long_puts.empty:
            return None
        long_put = long_puts.iloc[-1]
        
        # Calculate metrics
        call_credit = short_call['bid'] - long_call['ask']
        put_credit = short_put['bid'] - long_put['ask']
        net_credit = call_credit + put_credit
        
        call_spread_width = long_call['strike'] - short_call['strike']
        put_spread_width = short_put['strike'] - long_put['strike']
        max_loss = max(call_spread_width, put_spread_width) - net_credit
        
        # Position sizing based on max loss
        max_risk = account_size * self.max_position_size
        contracts = int(max_risk / (max_loss * 100))
        contracts = max(1, contracts)
        
        return OptionsPosition(
            strategy_type=StrategyType.IRON_CONDOR,
            legs=[
                {'contract': short_call['symbol'], 'strike': short_call['strike'], 'type': 'call', 
                 'action': 'sell', 'quantity': contracts, 'price': short_call['bid']},
                {'contract': long_call['symbol'], 'strike': long_call['strike'], 'type': 'call',
                 'action': 'buy', 'quantity': contracts, 'price': long_call['ask']},
                {'contract': short_put['symbol'], 'strike': short_put['strike'], 'type': 'put',
                 'action': 'sell', 'quantity': contracts, 'price': short_put['bid']},
                {'contract': long_put['symbol'], 'strike': long_put['strike'], 'type': 'put',
                 'action': 'buy', 'quantity': contracts, 'price': long_put['ask']}
            ],
            net_debit=0,
            net_credit=net_credit * contracts * 100,
            max_profit=net_credit * contracts * 100,
            max_loss=max_loss * contracts * 100,
            breakeven_points=[
                short_put['strike'] + net_credit,
                short_call['strike'] - net_credit
            ],
            probability_profit=self._calculate_condor_probability(short_put, short_call, spot),
            expected_value=self._calculate_condor_expected_value(
                short_put, short_call, long_put, long_call, spot, net_credit
            ),
            portfolio_greeks=self._calculate_condor_greeks(
                short_call, long_call, short_put, long_put, contracts
            ),
            risk_reward_ratio=net_credit / max_loss if max_loss > 0 else 0,
            margin_requirement=max_loss * contracts * 100
        )
    
    def _build_short_strangle(self, chain: pd.DataFrame, spot: float, account_size: float) -> OptionsPosition:
        """Build a short strangle for high IV environments."""
        
        calls = chain[chain['option_type'] == 'call'].sort_values('strike')
        puts = chain[chain['option_type'] == 'put'].sort_values('strike')
        
        # Select strikes at ~0.16 delta (1 SD move)
        target_delta = 0.16
        
        calls['delta_diff'] = abs(abs(calls['delta']) - target_delta)
        short_call = calls.nsmallest(1, 'delta_diff').iloc[0]
        
        puts['delta_diff'] = abs(abs(puts['delta']) - target_delta)
        short_put = puts.nsmallest(1, 'delta_diff').iloc[0]
        
        # Calculate credit and margin
        net_credit = short_call['bid'] + short_put['bid']
        
        # Margin approximation (simplified)
        margin_requirement = max(
            short_call['strike'] * 0.2,  # 20% of strike
            short_put['strike'] * 0.2
        ) + net_credit
        
        # Position sizing based on margin
        max_margin = account_size * 0.3  # Use max 30% of account for margin
        contracts = int(max_margin / (margin_requirement * 100))
        contracts = max(1, min(contracts, 5))  # Cap at 5 contracts for risk
        
        return OptionsPosition(
            strategy_type=StrategyType.SHORT_STRANGLE,
            legs=[
                {'contract': short_call['symbol'], 'strike': short_call['strike'], 'type': 'call',
                 'action': 'sell', 'quantity': contracts, 'price': short_call['bid']},
                {'contract': short_put['symbol'], 'strike': short_put['strike'], 'type': 'put',
                 'action': 'sell', 'quantity': contracts, 'price': short_put['bid']}
            ],
            net_debit=0,
            net_credit=net_credit * contracts * 100,
            max_profit=net_credit * contracts * 100,
            max_loss=float('inf'),  # Undefined risk
            breakeven_points=[
                short_put['strike'] - net_credit,
                short_call['strike'] + net_credit
            ],
            probability_profit=self._calculate_strangle_probability(short_put, short_call, spot),
            expected_value=net_credit * contracts * 100 * 0.7,  # Rough estimate
            portfolio_greeks=self._calculate_strangle_greeks(short_call, short_put, contracts),
            risk_reward_ratio=0,  # Undefined for unlimited risk
            margin_requirement=margin_requirement * contracts * 100
        )
    
    def _calculate_probability_profit(self, option: pd.Series, spot: float, option_type: str) -> float:
        """Calculate probability of profit for a single option."""
        
        # Simplified probability using delta as proxy
        if option_type == 'call':
            return abs(option['delta'])
        else:
            return 1 - abs(option['delta'])
    
    def _calculate_expected_value(self, option: pd.Series, spot: float, cost: float) -> float:
        """Calculate expected value of an option position."""
        
        # Simplified EV calculation
        prob_itm = abs(option['delta'])
        expected_intrinsic = (spot * 0.05) * prob_itm  # Assume 5% move if ITM
        return expected_intrinsic * 100 - cost
    
    def _calculate_spread_probability(self, long_leg: pd.Series, short_leg: pd.Series, spot: float) -> float:
        """Calculate probability of profit for a spread."""
        
        # Probability that spot is between strikes at expiration
        return abs(short_leg['delta']) - abs(long_leg['delta'])
    
    def _calculate_spread_expected_value(self, long_leg: pd.Series, short_leg: pd.Series, 
                                        spot: float, net_debit: float) -> float:
        """Calculate expected value of a spread."""
        
        prob_profit = self._calculate_spread_probability(long_leg, short_leg, spot)
        max_profit = short_leg['strike'] - long_leg['strike'] - net_debit
        return (prob_profit * max_profit - (1 - prob_profit) * net_debit) * 100
    
    def _calculate_condor_probability(self, short_put: pd.Series, short_call: pd.Series, spot: float) -> float:
        """Calculate probability of profit for iron condor."""
        
        # Probability that spot stays between short strikes
        return 1 - abs(short_put['delta']) - abs(short_call['delta'])
    
    def _calculate_condor_expected_value(self, short_put: pd.Series, short_call: pd.Series,
                                        long_put: pd.Series, long_call: pd.Series,
                                        spot: float, net_credit: float) -> float:
        """Calculate expected value of iron condor."""
        
        prob_profit = self._calculate_condor_probability(short_put, short_call, spot)
        return prob_profit * net_credit * 100
    
    def _calculate_strangle_probability(self, short_put: pd.Series, short_call: pd.Series, spot: float) -> float:
        """Calculate probability of profit for short strangle."""
        
        # Probability that spot stays between strikes + credit
        return 1 - abs(short_put['delta']) - abs(short_call['delta'])
    
    def _calculate_spread_greeks(self, long_leg: pd.Series, short_leg: pd.Series, contracts: int) -> Dict:
        """Calculate net Greeks for a spread."""
        
        return {
            'delta': contracts * 100 * (long_leg['delta'] - short_leg['delta']),
            'gamma': contracts * 100 * (long_leg['gamma'] - short_leg['gamma']),
            'theta': contracts * 100 * (long_leg['theta'] - short_leg['theta']),
            'vega': contracts * 100 * (long_leg['vega'] - short_leg['vega'])
        }
    
    def _calculate_condor_greeks(self, short_call: pd.Series, long_call: pd.Series,
                                short_put: pd.Series, long_put: pd.Series, contracts: int) -> Dict:
        """Calculate net Greeks for iron condor."""
        
        return {
            'delta': contracts * 100 * (
                long_call['delta'] - short_call['delta'] +
                long_put['delta'] - short_put['delta']
            ),
            'gamma': contracts * 100 * (
                long_call['gamma'] - short_call['gamma'] +
                long_put['gamma'] - short_put['gamma']
            ),
            'theta': contracts * 100 * (
                long_call['theta'] - short_call['theta'] +
                long_put['theta'] - short_put['theta']
            ),
            'vega': contracts * 100 * (
                long_call['vega'] - short_call['vega'] +
                long_put['vega'] - short_put['vega']
            )
        }
    
    def _calculate_strangle_greeks(self, short_call: pd.Series, short_put: pd.Series, contracts: int) -> Dict:
        """Calculate net Greeks for short strangle."""
        
        return {
            'delta': contracts * 100 * (-short_call['delta'] - short_put['delta']),
            'gamma': contracts * 100 * (-short_call['gamma'] - short_put['gamma']),
            'theta': contracts * 100 * (-short_call['theta'] - short_put['theta']),
            'vega': contracts * 100 * (-short_call['vega'] - short_put['vega'])
        }
    
    def optimize_portfolio_greeks(self, positions: List[OptionsPosition], 
                                 target_greeks: Dict[str, float]) -> List[Dict]:
        """
        Optimize portfolio Greeks by suggesting adjustments.
        
        Returns list of suggested trades to balance Greeks.
        """
        
        # Calculate current portfolio Greeks
        portfolio_greeks = {
            'delta': sum(pos.portfolio_greeks['delta'] for pos in positions),
            'gamma': sum(pos.portfolio_greeks['gamma'] for pos in positions),
            'theta': sum(pos.portfolio_greeks['theta'] for pos in positions),
            'vega': sum(pos.portfolio_greeks['vega'] for pos in positions)
        }
        
        adjustments = []
        
        # Delta adjustment
        if abs(portfolio_greeks['delta']) > self.max_portfolio_delta:
            delta_to_hedge = -portfolio_greeks['delta']
            adjustments.append({
                'type': 'delta_hedge',
                'amount': delta_to_hedge,
                'suggestion': f"Hedge {delta_to_hedge:.0f} deltas"
            })
        
        # Gamma adjustment
        if abs(portfolio_greeks['gamma']) > self.max_portfolio_gamma:
            adjustments.append({
                'type': 'gamma_hedge',
                'amount': -portfolio_greeks['gamma'],
                'suggestion': "Add calendar spreads or butterfly to reduce gamma"
            })
        
        # Vega adjustment
        if abs(portfolio_greeks['vega']) > self.max_portfolio_vega:
            adjustments.append({
                'type': 'vega_hedge',
                'amount': -portfolio_greeks['vega'],
                'suggestion': "Add opposing vega positions"
            })
        
        return adjustments