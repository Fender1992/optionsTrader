import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaVantageAPIError(Exception):
    pass

class AlphaVantageFetcher:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('ALPHAVANTAGE_API_KEY')
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not provided")
        
        self.base_url = "https://www.alphavantage.co/query"
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.HTTPError, AlphaVantageAPIError))
    )
    def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['apikey'] = self.api_key
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "Error Message" in data:
                raise AlphaVantageAPIError(f"API Error: {data['Error Message']}")
            if "Note" in data:
                logger.warning(f"API Note (possible rate limit): {data['Note']}")
                time.sleep(12)
                raise AlphaVantageAPIError("Rate limit hit, retrying...")
            if "Information" in data:
                logger.info(f"API Info: {data['Information']}")
                time.sleep(60)
                raise AlphaVantageAPIError("API limit reached, waiting...")
                
            return data
            
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise
    
    def fetch_daily_adjusted(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        cache_file = self.cache_dir / f"{symbol}_daily.parquet"
        
        existing_df = pd.DataFrame()
        if cache_file.exists():
            try:
                existing_df = pd.read_parquet(cache_file)
                logger.info(f"Loaded cached data for {symbol}: {len(existing_df)} rows")
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")
        
        params = {
            'function': 'TIME_SERIES_DAILY_ADJUSTED',
            'symbol': symbol,
            'outputsize': 'full'
        }
        
        logger.info(f"Fetching daily adjusted data for {symbol}")
        data = self._make_request(params)
        
        if "Time Series (Daily)" not in data:
            logger.error(f"No time series data for {symbol}")
            return pd.DataFrame()
        
        time_series = data["Time Series (Daily)"]
        
        rows = []
        for date_str, values in time_series.items():
            rows.append({
                'Date': pd.to_datetime(date_str),
                'Open': float(values['1. open']),
                'High': float(values['2. high']),
                'Low': float(values['3. low']),
                'Close': float(values['4. close']),
                'AdjClose': float(values['5. adjusted close']),
                'Volume': int(values['6. volume']),
                'Ticker': symbol
            })
        
        new_df = pd.DataFrame(rows)
        new_df = new_df.sort_values('Date')
        
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['Date', 'Ticker'], keep='last')
            combined_df = combined_df.sort_values('Date')
        else:
            combined_df = new_df
        
        if start_date:
            combined_df = combined_df[combined_df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            combined_df = combined_df[combined_df['Date'] <= pd.to_datetime(end_date)]
        
        combined_df.to_parquet(cache_file, index=False)
        logger.info(f"Saved {len(combined_df)} rows to cache for {symbol}")
        
        return combined_df
    
    def fetch_news_sentiment(self, symbols: List[str], start_date: str = None, end_date: str = None) -> pd.DataFrame:
        all_news = []
        
        for symbol in symbols:
            cache_file = self.cache_dir / f"{symbol}_news.parquet"
            
            existing_news = pd.DataFrame()
            if cache_file.exists():
                try:
                    existing_news = pd.read_parquet(cache_file)
                    logger.info(f"Loaded cached news for {symbol}: {len(existing_news)} articles")
                except Exception as e:
                    logger.warning(f"Failed to load news cache for {symbol}: {e}")
            
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'limit': 1000
            }
            
            if start_date:
                time_from = pd.to_datetime(start_date).strftime('%Y%m%dT%H%M')
                params['time_from'] = time_from
            
            if end_date:
                time_to = pd.to_datetime(end_date).strftime('%Y%m%dT%H%M')
                params['time_to'] = time_to
            
            logger.info(f"Fetching news sentiment for {symbol}")
            
            try:
                data = self._make_request(params)
            except Exception as e:
                logger.error(f"Failed to fetch news for {symbol}: {e}")
                if not existing_news.empty:
                    all_news.append(existing_news)
                continue
            
            if "feed" not in data:
                logger.warning(f"No news feed data for {symbol}")
                if not existing_news.empty:
                    all_news.append(existing_news)
                continue
            
            articles = data["feed"]
            
            for article in articles:
                ticker_sentiment = article.get("ticker_sentiment", [])
                
                for ticker_info in ticker_sentiment:
                    if ticker_info.get("ticker") == symbol:
                        all_news.append({
                            'published_time': pd.to_datetime(article.get("time_published", "")),
                            'ticker': symbol,
                            'title': article.get("title", ""),
                            'summary': article.get("summary", ""),
                            'overall_sentiment_score': float(article.get("overall_sentiment_score", 0)),
                            'overall_sentiment_label': article.get("overall_sentiment_label", ""),
                            'relevance_score': float(ticker_info.get("relevance_score", 0)),
                            'ticker_sentiment_score': float(ticker_info.get("ticker_sentiment_score", 0)),
                            'ticker_sentiment_label': ticker_info.get("ticker_sentiment_label", ""),
                            'source': article.get("source", ""),
                            'url': article.get("url", "")
                        })
            
            if all_news:
                news_df = pd.DataFrame(all_news)
                
                if not existing_news.empty:
                    combined_news = pd.concat([existing_news, news_df], ignore_index=True)
                    combined_news = combined_news.drop_duplicates(
                        subset=['published_time', 'ticker', 'title'], 
                        keep='last'
                    )
                else:
                    combined_news = news_df
                
                combined_news = combined_news.sort_values('published_time', ascending=False)
                combined_news.to_parquet(cache_file, index=False)
                logger.info(f"Saved {len(combined_news)} news articles for {symbol}")
        
        if all_news:
            return pd.DataFrame(all_news)
        return pd.DataFrame()
    
    def fetch_options_chain(self, symbol: str) -> pd.DataFrame:
        logger.warning("Alpha Vantage does not provide options chain data. Use Tradier API instead.")
        return pd.DataFrame()
    
    def fetch_company_overview(self, symbol: str) -> Dict[str, Any]:
        params = {
            'function': 'OVERVIEW',
            'symbol': symbol
        }
        
        logger.info(f"Fetching company overview for {symbol}")
        data = self._make_request(params)
        
        return data