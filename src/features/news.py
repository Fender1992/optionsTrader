import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NewsFeatureBuilder:
    def __init__(self, config: Dict):
        self.config = config.get('news', {})
        self.min_confidence = self.config.get('min_confidence', 0.55)
        self.sentiment_window = self.config.get('sentiment_window', 5)
        self.exclude_earnings_window = self.config.get('exclude_earnings_window', {})
        
    def build_news_features(self, news_df: pd.DataFrame, price_calendar: pd.DataFrame) -> pd.DataFrame:
        if news_df.empty:
            logger.warning("Empty news dataframe provided")
            return self._create_empty_features(price_calendar)
        
        news_df = news_df[news_df['relevance_score'] >= self.min_confidence].copy()
        
        if news_df.empty:
            logger.warning(f"No news articles meet minimum confidence threshold {self.min_confidence}")
            return self._create_empty_features(price_calendar)
        
        news_df['published_date'] = pd.to_datetime(news_df['published_time']).dt.date
        news_df['published_date'] = pd.to_datetime(news_df['published_date'])
        
        daily_agg = news_df.groupby(['ticker', 'published_date']).agg({
            'overall_sentiment_score': ['mean', 'median', 'std'],
            'ticker_sentiment_score': ['mean', 'median', 'std'],
            'relevance_score': 'mean',
            'title': 'count',
            'source': lambda x: len(x.unique())
        }).reset_index()
        
        daily_agg.columns = [
            'ticker', 'date',
            'sent_overall_mean', 'sent_overall_median', 'sent_overall_std',
            'sent_ticker_mean', 'sent_ticker_median', 'sent_ticker_std',
            'relevance_mean',
            'article_count',
            'source_diversity'
        ]
        
        features_list = []
        
        for ticker in daily_agg['ticker'].unique():
            ticker_news = daily_agg[daily_agg['ticker'] == ticker].sort_values('date')
            ticker_calendar = price_calendar[price_calendar['Ticker'] == ticker].copy()
            
            if ticker_calendar.empty:
                continue
            
            ticker_calendar = ticker_calendar.sort_values('Date')
            
            merged = pd.merge_asof(
                ticker_calendar,
                ticker_news,
                left_on='Date',
                right_on='date',
                direction='backward',
                tolerance=pd.Timedelta(days=1)
            )
            
            merged['ticker'] = ticker
            
            for col in ['sent_overall_mean', 'sent_ticker_mean', 'article_count']:
                if col in merged.columns:
                    merged[f'{col}_1d'] = merged[col].fillna(0)
                    merged[f'{col}_3d'] = merged[col].rolling(window=3, min_periods=1).mean()
                    merged[f'{col}_5d'] = merged[col].rolling(window=5, min_periods=1).mean()
                    merged[f'{col}_10d'] = merged[col].rolling(window=10, min_periods=1).mean()
            
            merged['sent_momentum'] = merged['sent_ticker_mean_3d'] - merged['sent_ticker_mean_10d']
            
            merged['sent_volatility_5d'] = merged['sent_ticker_mean'].rolling(window=5, min_periods=1).std()
            
            merged['news_volume_spike'] = merged['article_count'] / (merged['article_count'].rolling(window=20, min_periods=1).mean() + 1)
            
            extreme_neg_threshold = -0.3
            merged['extreme_negative_news'] = (merged['sent_ticker_mean_1d'] < extreme_neg_threshold).astype(int)
            
            extreme_pos_threshold = 0.3
            merged['extreme_positive_news'] = (merged['sent_ticker_mean_1d'] > extreme_pos_threshold).astype(int)
            
            features_list.append(merged)
        
        if features_list:
            all_features = pd.concat(features_list, ignore_index=True)
            
            feature_cols = [
                'Date', 'ticker',
                'sent_overall_mean_1d', 'sent_overall_mean_3d', 'sent_overall_mean_5d',
                'sent_ticker_mean_1d', 'sent_ticker_mean_3d', 'sent_ticker_mean_5d', 'sent_ticker_mean_10d',
                'article_count_1d', 'article_count_3d', 'article_count_5d',
                'sent_momentum', 'sent_volatility_5d', 'news_volume_spike',
                'extreme_negative_news', 'extreme_positive_news',
                'source_diversity'
            ]
            
            available_cols = [col for col in feature_cols if col in all_features.columns]
            return all_features[available_cols]
        
        return self._create_empty_features(price_calendar)
    
    def _create_empty_features(self, price_calendar: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            'sent_overall_mean_1d', 'sent_overall_mean_3d', 'sent_overall_mean_5d',
            'sent_ticker_mean_1d', 'sent_ticker_mean_3d', 'sent_ticker_mean_5d', 'sent_ticker_mean_10d',
            'article_count_1d', 'article_count_3d', 'article_count_5d',
            'sent_momentum', 'sent_volatility_5d', 'news_volume_spike',
            'extreme_negative_news', 'extreme_positive_news',
            'source_diversity'
        ]
        
        result = price_calendar[['Date', 'Ticker']].copy()
        result.rename(columns={'Ticker': 'ticker'}, inplace=True)
        
        for col in feature_cols:
            result[col] = 0
        
        return result
    
    def detect_earnings_window(self, news_df: pd.DataFrame, ticker: str, date: pd.datetime) -> bool:
        if news_df.empty:
            return False
        
        before_days = self.exclude_earnings_window.get('before', 1)
        after_days = self.exclude_earnings_window.get('after', 1)
        
        ticker_news = news_df[news_df['ticker'] == ticker].copy()
        
        earnings_keywords = ['earnings', 'quarterly results', 'q1', 'q2', 'q3', 'q4', 'eps', 'revenue beat', 'revenue miss']
        
        window_start = date - timedelta(days=before_days)
        window_end = date + timedelta(days=after_days)
        
        window_news = ticker_news[
            (ticker_news['published_time'] >= window_start) &
            (ticker_news['published_time'] <= window_end)
        ]
        
        for _, article in window_news.iterrows():
            title_lower = article['title'].lower() if pd.notna(article['title']) else ''
            summary_lower = article['summary'].lower() if pd.notna(article['summary']) else ''
            
            if any(keyword in title_lower or keyword in summary_lower for keyword in earnings_keywords):
                return True
        
        return False