#!/usr/bin/env python
"""
Alpha Vantage Data Fetcher
Simplified for maximum profit trading system
"""

import asyncio
import aiohttp
import pandas as pd
import logging
import os
from typing import Optional, Dict
import json

logger = logging.getLogger(__name__)

class AlphaVantageDataFetcher:
    """Simplified Alpha Vantage data fetcher for high-frequency trading"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ALPHAVANTAGE_API_KEY', os.getenv('ALPHA_VANTAGE_API_KEY'))
        if not self.api_key:
            raise ValueError("Alpha Vantage API key is required. Set ALPHA_VANTAGE_API_KEY environment variable.")
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def fetch_intraday_data(self, symbol: str, interval: str = '1min') -> Optional[pd.DataFrame]:
        """Fetch real intraday data for a symbol from Alpha Vantage"""
        
        try:
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(self.base_url, params=params) as response:
                data = await response.json()
                
                # Handle API errors
                if 'Error Message' in data:
                    logger.error(f"API Error for {symbol}: {data['Error Message']}")
                    return None
                    
                if 'Note' in data:
                    logger.warning(f"API Rate limit for {symbol}: {data['Note']}")
                    return None
                
                # Extract time series data
                ts_key = f'Time Series ({interval})'
                if ts_key not in data:
                    logger.error(f"No time series data for {symbol}")
                    return None
                
                # Convert to DataFrame
                ts_data = data[ts_key]
                df = pd.DataFrame.from_dict(ts_data, orient='index')
                
                # Clean column names and types
                df.columns = ['open', 'high', 'low', 'close', 'volume']
                df.index = pd.to_datetime(df.index)
                df = df.astype(float)
                df = df.sort_index()
                
                logger.debug(f"Fetched {len(df)} bars for {symbol}")
                return df
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    async def fetch_news_sentiment(self, symbol: str) -> Optional[Dict]:
        """Fetch real news sentiment from Alpha Vantage"""
        
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.api_key,
                'limit': '20'
            }
            
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(self.base_url, params=params) as response:
                data = await response.json()
                
                if 'feed' not in data:
                    return None
                
                # Aggregate sentiment
                articles = data['feed']
                if not articles:
                    return None
                
                # Calculate average sentiment
                sentiment_scores = []
                for article in articles[:10]:  # Use latest 10 articles
                    ticker_sentiments = article.get('ticker_sentiment', [])
                    for ts in ticker_sentiments:
                        if ts.get('ticker') == symbol:
                            sentiment_scores.append(float(ts.get('relevance_score', 0)))
                
                if sentiment_scores:
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                    
                    # Convert to simple classification
                    if avg_sentiment > 0.1:
                        sentiment = 'bullish'
                    elif avg_sentiment < -0.1:
                        sentiment = 'bearish'
                    else:
                        sentiment = 'neutral'
                    
                    return {
                        'sentiment': sentiment,
                        'score': avg_sentiment,
                        'articles_count': len(articles)
                    }
                
        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return None
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()