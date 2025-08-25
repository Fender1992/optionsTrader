#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml
from datetime import datetime, timedelta
import logging
from src.fetchers.alpha_vantage import AlphaVantageFetcher
from src.features.technical import TechnicalFeatureBuilder
from src.features.news import NewsFeatureBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    fetcher = AlphaVantageFetcher()
    technical_builder = TechnicalFeatureBuilder(config)
    news_builder = NewsFeatureBuilder(config)
    
    universe_file = config.get('universe_file', 'data/symbols.csv')
    if not os.path.exists(universe_file):
        symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
        pd.DataFrame({'symbol': symbols}).to_csv(universe_file, index=False)
    else:
        symbols = pd.read_csv(universe_file)['symbol'].tolist()
    
    logger.info(f"Building features for {len(symbols)} symbols")
    
    start_date = config.get('start_date', '2014-01-01')
    end_date = config.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    
    all_price_data = []
    for symbol in symbols:
        try:
            logger.info(f"Fetching price data for {symbol}")
            price_df = fetcher.fetch_daily_adjusted(symbol, start_date, end_date)
            
            if not price_df.empty:
                all_price_data.append(price_df)
                logger.info(f"Fetched {len(price_df)} rows for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
    
    if not all_price_data:
        logger.error("No price data fetched")
        return
    
    price_df = pd.concat(all_price_data, ignore_index=True)
    logger.info(f"Combined price data: {len(price_df)} rows")
    
    logger.info("Building technical features")
    technical_features = technical_builder.build_features(price_df)
    
    logger.info("Fetching news data")
    news_lookback = config.get('news', {}).get('lookback_days', 30)
    news_start = (datetime.now() - timedelta(days=news_lookback)).strftime('%Y-%m-%d')
    
    news_df = fetcher.fetch_news_sentiment(symbols, news_start, end_date)
    
    if not news_df.empty:
        logger.info(f"Fetched {len(news_df)} news articles")
        news_features = news_builder.build_news_features(news_df, price_df)
    else:
        logger.warning("No news data available")
        news_features = news_builder._create_empty_features(price_df)
    
    if not technical_features.empty and not news_features.empty:
        features = pd.merge(
            technical_features,
            news_features,
            left_on=['Date', 'Ticker'],
            right_on=['Date', 'ticker'],
            how='left'
        )
        
        if 'ticker' in features.columns:
            features = features.drop('ticker', axis=1)
    else:
        features = technical_features
    
    os.makedirs('data', exist_ok=True)
    features.to_parquet('data/features.parquet', index=False)
    
    logger.info(f"Features saved: {len(features)} rows, {len(features.columns)} columns")
    
    sample_features = features.columns.tolist()[:20]
    logger.info(f"Sample features: {sample_features}")

if __name__ == "__main__":
    main()