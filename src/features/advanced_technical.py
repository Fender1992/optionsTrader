import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from numpy import nan

logger = logging.getLogger(__name__)

class AdvancedTechnicalIndicators:
    """
    Advanced technical indicators module with 60+ metrics for options trading.
    Includes momentum, volatility, volume, trend, and statistical indicators.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.lookback_periods = [5, 10, 20, 50, 100, 200]
        
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given dataframe."""
        if df.empty:
            logger.warning("Empty dataframe provided for technical analysis")
            return df
        
        df = df.sort_values(['Ticker', 'Date']).copy()
        results = []
        
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('Date')
            
            # Price-based features
            ticker_df = self._calculate_returns(ticker_df)
            ticker_df = self._calculate_moving_averages(ticker_df)
            ticker_df = self._calculate_momentum_indicators(ticker_df)
            ticker_df = self._calculate_volatility_indicators(ticker_df)
            ticker_df = self._calculate_volume_indicators(ticker_df)
            ticker_df = self._calculate_trend_indicators(ticker_df)
            ticker_df = self._calculate_oscillators(ticker_df)
            ticker_df = self._calculate_statistical_indicators(ticker_df)
            ticker_df = self._calculate_market_regime_indicators(ticker_df)
            ticker_df = self._calculate_options_relevant_indicators(ticker_df)
            
            results.append(ticker_df)
        
        if results:
            result_df = pd.concat(results, ignore_index=True)
            result_df = result_df.replace([np.inf, -np.inf], np.nan)
            return result_df
        
        return df
    
    def _calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various return metrics."""
        # Simple returns
        for period in [1, 5, 10, 21, 63, 126, 252]:
            df[f'Return_{period}d'] = df['AdjClose'].pct_change(periods=period)
            df[f'LogReturn_{period}d'] = np.log(df['AdjClose'] / df['AdjClose'].shift(period))
        
        # Overnight and intraday returns
        df['Overnight_Return'] = df['Open'] / df['AdjClose'].shift(1) - 1
        df['Intraday_Return'] = df['AdjClose'] / df['Open'] - 1
        
        # Gap analysis
        df['Gap'] = df['Open'] / df['Close'].shift(1) - 1
        df['Gap_MA_5'] = df['Gap'].rolling(window=5).mean()
        df['Gap_Std_5'] = df['Gap'].rolling(window=5).std()
        
        return df
    
    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages and crossovers."""
        # Simple Moving Averages
        for period in self.lookback_periods:
            df[f'SMA_{period}'] = df['AdjClose'].rolling(window=period).mean()
            df[f'Price_to_SMA_{period}'] = df['AdjClose'] / df[f'SMA_{period}']
        
        # Exponential Moving Averages
        for period in [12, 26, 50, 200]:
            df[f'EMA_{period}'] = df['AdjClose'].ewm(span=period, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # Hull Moving Average
        for period in [20, 50]:
            half_period = period // 2
            sqrt_period = int(np.sqrt(period))
            wma_half = df['AdjClose'].rolling(window=half_period).apply(
                lambda x: np.sum(x * np.arange(1, half_period + 1)) / np.sum(np.arange(1, half_period + 1))
            )
            wma_full = df['AdjClose'].rolling(window=period).apply(
                lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1))
            )
            df[f'HMA_{period}'] = (2 * wma_half - wma_full).rolling(window=sqrt_period).mean()
        
        # KAMA (Kaufman Adaptive Moving Average)
        df['KAMA'] = ta.kama(df['AdjClose'], length=10)
        
        # TEMA (Triple Exponential Moving Average)
        df['TEMA'] = ta.tema(df['AdjClose'], length=20)
        
        # Moving Average Crossovers
        df['Golden_Cross'] = (df['SMA_50'] > df['SMA_200']).astype(int)
        df['Death_Cross'] = (df['SMA_50'] < df['SMA_200']).astype(int)
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators."""
        # RSI (Relative Strength Index)
        for period in [7, 14, 21]:
            df[f'RSI_{period}'] = ta.rsi(df['AdjClose'], length=period)
        
        # Stochastic Oscillator
        for period in [14, 21]:
            stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=period, d=3)
            if stoch is not None and not stoch.empty:
                df[f'Stoch_K_{period}'] = stoch[f'STOCHk_{period}_3_3']
                df[f'Stoch_D_{period}'] = stoch[f'STOCHd_{period}_3_3']
        
        # Williams %R
        for period in [14, 28]:
            df[f'Williams_R_{period}'] = ta.willr(df['High'], df['Low'], df['Close'], length=period)
        
        # ROC (Rate of Change)
        for period in [10, 20, 30]:
            df[f'ROC_{period}'] = ta.roc(df['AdjClose'], length=period)
        
        # Momentum
        for period in [10, 20, 40]:
            df[f'Momentum_{period}'] = df['AdjClose'] / df['AdjClose'].shift(period) - 1
        
        # TSI (True Strength Index)
        df['TSI'] = ta.tsi(df['AdjClose'])
        
        # PPO (Percentage Price Oscillator)
        df['PPO'] = ta.ppo(df['AdjClose'])
        
        # TRIX
        df['TRIX'] = ta.trix(df['AdjClose'])
        
        return df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators."""
        # Historical Volatility
        for period in [5, 10, 20, 60]:
            df[f'Volatility_{period}'] = df['LogReturn_1d'].rolling(window=period).std() * np.sqrt(252)
        
        # ATR (Average True Range)
        for period in [14, 20]:
            df[f'ATR_{period}'] = ta.atr(df['High'], df['Low'], df['Close'], length=period)
            df[f'ATR_Ratio_{period}'] = df[f'ATR_{period}'] / df['AdjClose']
        
        # Bollinger Bands
        for period in [20, 50]:
            bb = ta.bbands(df['AdjClose'], length=period, std=2)
            if bb is not None and not bb.empty:
                df[f'BB_Upper_{period}'] = bb[f'BBU_{period}_2.0']
                df[f'BB_Middle_{period}'] = bb[f'BBM_{period}_2.0']
                df[f'BB_Lower_{period}'] = bb[f'BBL_{period}_2.0']
                df[f'BB_Width_{period}'] = (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}']) / df[f'BB_Middle_{period}']
                df[f'BB_Position_{period}'] = (df['AdjClose'] - df[f'BB_Lower_{period}']) / (df[f'BB_Upper_{period}'] - df[f'BB_Lower_{period}'])
        
        # Keltner Channels
        kc = ta.kc(df['High'], df['Low'], df['Close'], length=20)
        if kc is not None and not kc.empty:
            df['KC_Upper'] = kc['KCUe_20_2']
            df['KC_Lower'] = kc['KCLe_20_2']
            df['KC_Position'] = (df['AdjClose'] - df['KC_Lower']) / (df['KC_Upper'] - df['KC_Lower'])
        
        # Donchian Channels
        for period in [20, 55]:
            df[f'DC_Upper_{period}'] = df['High'].rolling(window=period).max()
            df[f'DC_Lower_{period}'] = df['Low'].rolling(window=period).min()
            df[f'DC_Middle_{period}'] = (df[f'DC_Upper_{period}'] + df[f'DC_Lower_{period}']) / 2
            df[f'DC_Position_{period}'] = (df['AdjClose'] - df[f'DC_Lower_{period}']) / (df[f'DC_Upper_{period}'] - df[f'DC_Lower_{period}'])
        
        # Garman-Klass Volatility
        df['GK_Volatility'] = np.sqrt(
            252 * df.apply(lambda x: 0.5 * np.log(x['High']/x['Low'])**2 - 
                          (2*np.log(2)-1) * np.log(x['Close']/x['Open'])**2, axis=1).rolling(window=20).mean()
        )
        
        # Parkinson Volatility
        df['Parkinson_Vol'] = np.sqrt(252 / (4 * np.log(2)) * (np.log(df['High']/df['Low'])**2).rolling(window=20).mean())
        
        return df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        # Volume Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'Volume_MA_{period}'] = df['Volume'].rolling(window=period).mean()
            df[f'Volume_Ratio_{period}'] = df['Volume'] / df[f'Volume_MA_{period}']
        
        # OBV (On Balance Volume)
        df['OBV'] = (np.sign(df['Returns']) * df['Volume']).cumsum()
        df['OBV_MA_20'] = df['OBV'].rolling(window=20).mean()
        
        # CMF (Chaikin Money Flow)
        df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # MFI (Money Flow Index)
        df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # ADL (Accumulation/Distribution Line)
        df['ADL'] = ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # VWAP (Volume Weighted Average Price)
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['Price_to_VWAP'] = df['AdjClose'] / df['VWAP']
        
        # Volume Price Trend
        df['VPT'] = (df['Volume'] * df['Returns']).cumsum()
        
        # Ease of Movement
        df['EOM'] = ta.eom(df['High'], df['Low'], df['Volume'], df['Close'])
        
        # Force Index
        df['Force_Index'] = df['Volume'] * df['Returns']
        df['Force_Index_MA_13'] = df['Force_Index'].rolling(window=13).mean()
        
        return df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators."""
        # ADX (Average Directional Index)
        adx = ta.adx(df['High'], df['Low'], df['Close'])
        if adx is not None and not adx.empty:
            df['ADX'] = adx['ADX_14']
            df['DI_Plus'] = adx['DMP_14']
            df['DI_Minus'] = adx['DMN_14']
        
        # Aroon
        aroon = ta.aroon(df['High'], df['Low'])
        if aroon is not None and not aroon.empty:
            df['Aroon_Up'] = aroon['AROONU_14']
            df['Aroon_Down'] = aroon['AROOND_14']
            df['Aroon_Oscillator'] = df['Aroon_Up'] - df['Aroon_Down']
        
        # Parabolic SAR
        df['PSAR'] = ta.psar(df['High'], df['Low'], df['Close'])['PSARl_0.02_0.2']
        df['PSAR_Signal'] = (df['AdjClose'] > df['PSAR']).astype(int)
        
        # Supertrend
        st = ta.supertrend(df['High'], df['Low'], df['Close'])
        if st is not None and not st.empty:
            df['Supertrend'] = st['SUPERT_7_3.0']
            df['Supertrend_Signal'] = (df['AdjClose'] > df['Supertrend']).astype(int)
        
        # Ichimoku Cloud
        ich = ta.ichimoku(df['High'], df['Low'], df['Close'])
        if ich is not None and not ich.empty and len(ich.columns) > 0:
            df['Ichimoku_Conversion'] = ich[0]['ITS_9']
            df['Ichimoku_Base'] = ich[0]['IKS_26']
            df['Ichimoku_SpanA'] = ich[0]['ISA_9']
            df['Ichimoku_SpanB'] = ich[0]['ISB_26']
        
        return df
    
    def _calculate_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate oscillator indicators."""
        # CCI (Commodity Channel Index)
        df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
        
        # Ultimate Oscillator
        df['Ultimate_Oscillator'] = ta.uo(df['High'], df['Low'], df['Close'])
        
        # Awesome Oscillator
        df['Awesome_Oscillator'] = ta.ao(df['High'], df['Low'])
        
        # MACD Percentage
        if 'MACD' in df.columns and 'MACD_Signal' in df.columns:
            df['MACD_Percentage'] = (df['MACD'] - df['MACD_Signal']) / df['AdjClose'] * 100
        
        # Stochastic RSI
        for period in [14, 21]:
            if f'RSI_{period}' in df.columns:
                df[f'Stoch_RSI_{period}'] = ta.stochrsi(df['AdjClose'], length=period)['STOCHRSIk_14_14_3_3']
        
        return df
    
    def _calculate_statistical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical indicators."""
        # Rolling Statistics
        for period in [20, 50]:
            # Skewness and Kurtosis
            df[f'Skew_{period}'] = df['LogReturn_1d'].rolling(window=period).skew()
            df[f'Kurtosis_{period}'] = df['LogReturn_1d'].rolling(window=period).kurt()
            
            # Z-Score
            df[f'Z_Score_{period}'] = (df['AdjClose'] - df[f'SMA_{period}']) / df['LogReturn_1d'].rolling(window=period).std()
            
            # Percentile Rank
            df[f'Percentile_Rank_{period}'] = df['AdjClose'].rolling(window=period).rank(pct=True)
        
        # Maximum Drawdown
        for period in [20, 50, 100]:
            rolling_max = df['AdjClose'].rolling(window=period).max()
            df[f'Drawdown_{period}'] = (df['AdjClose'] - rolling_max) / rolling_max
            df[f'Max_Drawdown_{period}'] = df[f'Drawdown_{period}'].rolling(window=period).min()
        
        # Sharpe Ratio (rolling)
        for period in [20, 60]:
            df[f'Sharpe_{period}'] = (df['LogReturn_1d'].rolling(window=period).mean() * 252) / (df['LogReturn_1d'].rolling(window=period).std() * np.sqrt(252))
        
        # Sortino Ratio (rolling)
        for period in [20, 60]:
            downside_returns = df['LogReturn_1d'].copy()
            downside_returns[downside_returns > 0] = 0
            downside_vol = downside_returns.rolling(window=period).std() * np.sqrt(252)
            df[f'Sortino_{period}'] = (df['LogReturn_1d'].rolling(window=period).mean() * 252) / downside_vol
        
        # Calmar Ratio
        for period in [252]:
            annual_return = df['LogReturn_1d'].rolling(window=period).mean() * 252
            df[f'Calmar_{period}'] = annual_return / abs(df[f'Max_Drawdown_{period}'])
        
        return df
    
    def _calculate_market_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate market regime indicators."""
        # Trend Strength
        df['Trend_Strength'] = abs(df['SMA_20'] - df['SMA_50']) / df['SMA_50']
        
        # Volatility Regime
        df['Vol_Regime'] = df['Volatility_20'] / df['Volatility_60'].rolling(window=252).mean()
        
        # Market Phase (based on SMA positions)
        conditions = [
            (df['AdjClose'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_200']),
            (df['AdjClose'] < df['SMA_50']) & (df['SMA_50'] > df['SMA_200']),
            (df['AdjClose'] > df['SMA_50']) & (df['SMA_50'] < df['SMA_200']),
            (df['AdjClose'] < df['SMA_50']) & (df['SMA_50'] < df['SMA_200'])
        ]
        choices = [3, 2, 1, 0]  # Bull, Distribution, Accumulation, Bear
        df['Market_Phase'] = np.select(conditions, choices, default=2)
        
        # Momentum Quality
        df['Momentum_Quality'] = df['Momentum_20'] / df['Volatility_20']
        
        # Trend Consistency
        df['Trend_Consistency'] = df['LogReturn_1d'].rolling(window=20).apply(lambda x: sum(x > 0) / len(x))
        
        return df
    
    def _calculate_options_relevant_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators specifically relevant for options trading."""
        # Volatility Term Structure Proxy (using different period volatilities)
        df['Vol_Term_Structure'] = df['Volatility_10'] / df['Volatility_60']
        
        # Volatility of Volatility
        df['Vol_of_Vol'] = df['Volatility_20'].rolling(window=20).std()
        
        # Price Range Indicators (important for options strikes)
        for period in [5, 10, 20]:
            df[f'High_Low_Range_{period}'] = (df['High'].rolling(window=period).max() - 
                                              df['Low'].rolling(window=period).min()) / df['AdjClose']
            df[f'True_Range_{period}'] = df[f'ATR_{min(period, 20)}'] / df['AdjClose'] if f'ATR_{min(period, 20)}' in df.columns else 0
        
        # Gamma Proxy (rate of change of delta ~ second derivative of price)
        df['Price_Acceleration'] = df['Returns'].diff()
        df['Price_Acceleration_MA_5'] = df['Price_Acceleration'].rolling(window=5).mean()
        
        # Mean Reversion Indicators (for premium selling strategies)
        for period in [20, 50]:
            df[f'Mean_Reversion_{period}'] = (df['AdjClose'] - df[f'SMA_{period}']) / df[f'SMA_{period}']
            df[f'Mean_Reversion_Z_{period}'] = df[f'Mean_Reversion_{period}'] / df[f'Mean_Reversion_{period}'].rolling(window=period).std()
        
        # Support/Resistance Levels (for strike selection)
        for period in [20, 50]:
            df[f'Resistance_{period}'] = df['High'].rolling(window=period).max()
            df[f'Support_{period}'] = df['Low'].rolling(window=period).min()
            df[f'Price_to_Resistance_{period}'] = df['AdjClose'] / df[f'Resistance_{period}']
            df[f'Price_to_Support_{period}'] = df['AdjClose'] / df[f'Support_{period}']
        
        # Options Sentiment Proxies
        df['Put_Call_Skew_Proxy'] = df['Volatility_10'] * df['Mean_Reversion_20'] if 'Mean_Reversion_20' in df.columns else 0
        
        # Event Risk Proxy (sudden volatility spikes)
        df['Vol_Spike'] = df['Volatility_5'] / df['Volatility_20'].rolling(window=60).median()
        
        # Time Decay Favorable Conditions (high vol, low realized movement)
        df['Theta_Favorable'] = (df['Volatility_20'] > df['Volatility_60']) & (abs(df['Return_5d']) < df['Volatility_5'] * np.sqrt(5/252))
        
        return df
    
    def get_feature_importance_hints(self) -> Dict[str, float]:
        """Return expected feature importance for options trading."""
        return {
            'Volatility_20': 0.15,
            'RSI_14': 0.12,
            'Vol_Term_Structure': 0.10,
            'Mean_Reversion_Z_20': 0.08,
            'MACD_Histogram': 0.07,
            'Volume_Ratio_20': 0.06,
            'ATR_Ratio_14': 0.05,
            'Momentum_Quality': 0.05,
            'BB_Position_20': 0.04,
            'ADX': 0.04,
            'Vol_Spike': 0.03,
            'Market_Phase': 0.03,
            'Theta_Favorable': 0.03
        }