import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalFeatureBuilder:
    def __init__(self, config: Dict):
        self.config = config
        
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            logger.warning("Empty dataframe provided for technical features")
            return df
        
        df = df.sort_values(['Ticker', 'Date']).copy()
        
        features = []
        
        for ticker in df['Ticker'].unique():
            ticker_df = df[df['Ticker'] == ticker].copy()
            ticker_df = ticker_df.sort_values('Date')
            
            ticker_df['Returns'] = ticker_df['AdjClose'].pct_change()
            
            for period in [5, 10, 20, 50, 200]:
                ticker_df[f'SMA_{period}'] = ticker_df['AdjClose'].rolling(window=period).mean()
                ticker_df[f'Price_to_SMA_{period}'] = ticker_df['AdjClose'] / ticker_df[f'SMA_{period}']
            
            for period in [12, 26]:
                ticker_df[f'EMA_{period}'] = ticker_df['AdjClose'].ewm(span=period, adjust=False).mean()
            
            ticker_df['MACD'] = ticker_df['EMA_12'] - ticker_df['EMA_26']
            ticker_df['MACD_Signal'] = ticker_df['MACD'].ewm(span=9, adjust=False).mean()
            ticker_df['MACD_Histogram'] = ticker_df['MACD'] - ticker_df['MACD_Signal']
            
            ticker_df['RSI_14'] = ta.rsi(ticker_df['AdjClose'], length=14)
            ticker_df['RSI_7'] = ta.rsi(ticker_df['AdjClose'], length=7)
            
            bb = ta.bbands(ticker_df['AdjClose'], length=20, std=2)
            if bb is not None and not bb.empty:
                ticker_df['BB_Upper'] = bb['BBU_20_2.0']
                ticker_df['BB_Middle'] = bb['BBM_20_2.0']
                ticker_df['BB_Lower'] = bb['BBL_20_2.0']
                ticker_df['BB_Width'] = (ticker_df['BB_Upper'] - ticker_df['BB_Lower']) / ticker_df['BB_Middle']
                ticker_df['BB_Position'] = (ticker_df['AdjClose'] - ticker_df['BB_Lower']) / (ticker_df['BB_Upper'] - ticker_df['BB_Lower'])
            
            ticker_df['ATR_14'] = ta.atr(ticker_df['High'], ticker_df['Low'], ticker_df['Close'], length=14)
            
            for period in [5, 10, 20]:
                ticker_df[f'Volume_MA_{period}'] = ticker_df['Volume'].rolling(window=period).mean()
                ticker_df[f'Volume_Ratio_{period}'] = ticker_df['Volume'] / ticker_df[f'Volume_MA_{period}']
            
            ticker_df['Volume_Price_Trend'] = (ticker_df['Volume'] * ticker_df['Returns']).cumsum()
            
            ticker_df['OBV'] = (np.sign(ticker_df['Returns']) * ticker_df['Volume']).cumsum()
            
            for period in [5, 10, 20]:
                ticker_df[f'Volatility_{period}'] = ticker_df['Returns'].rolling(window=period).std() * np.sqrt(252)
            
            ticker_df['High_Low_Ratio'] = ticker_df['High'] / ticker_df['Low']
            ticker_df['Close_Open_Ratio'] = ticker_df['Close'] / ticker_df['Open']
            
            ticker_df['Gap'] = ticker_df['Open'] / ticker_df['Close'].shift(1) - 1
            ticker_df['Gap_MA_5'] = ticker_df['Gap'].rolling(window=5).mean()
            
            for period in [5, 10, 20, 50]:
                ticker_df[f'Return_{period}d'] = ticker_df['AdjClose'].pct_change(periods=period)
            
            ticker_df['Momentum_10'] = ticker_df['AdjClose'] / ticker_df['AdjClose'].shift(10) - 1
            ticker_df['Momentum_20'] = ticker_df['AdjClose'] / ticker_df['AdjClose'].shift(20) - 1
            
            high_20 = ticker_df['High'].rolling(window=20).max()
            low_20 = ticker_df['Low'].rolling(window=20).min()
            ticker_df['Stochastic_K'] = 100 * (ticker_df['Close'] - low_20) / (high_20 - low_20)
            ticker_df['Stochastic_D'] = ticker_df['Stochastic_K'].rolling(window=3).mean()
            
            ticker_df['Price_52W_High'] = ticker_df['High'].rolling(window=252).max()
            ticker_df['Price_52W_Low'] = ticker_df['Low'].rolling(window=252).min()
            ticker_df['Price_to_52W_High'] = ticker_df['AdjClose'] / ticker_df['Price_52W_High']
            ticker_df['Price_to_52W_Low'] = ticker_df['AdjClose'] / ticker_df['Price_52W_Low']
            
            features.append(ticker_df)
        
        if features:
            result = pd.concat(features, ignore_index=True)
            result = result.sort_values(['Ticker', 'Date'])
            
            result = result.replace([np.inf, -np.inf], np.nan)
            
            return result
        
        return df