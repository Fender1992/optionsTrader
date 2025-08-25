import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
import lightgbm as lgb
import xgboost as xgb
import joblib
import os
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class MLModel:
    def __init__(self, config: Dict):
        self.config = config
        self.model_type = 'lightgbm'
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.model_path = "artifacts/models/options_model.pkl"
        self.scaler_path = "artifacts/models/scaler.pkl"
        
        os.makedirs("artifacts/models", exist_ok=True)
        
        if os.path.exists(self.model_path):
            self.load_model()
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        feature_cols = [
            'Returns', 'RSI_14', 'RSI_7', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'Price_to_SMA_20', 'Price_to_SMA_50', 'Price_to_SMA_200',
            'BB_Position', 'BB_Width', 'ATR_14',
            'Volume_Ratio_5', 'Volume_Ratio_10', 'Volume_Ratio_20',
            'Volatility_5', 'Volatility_10', 'Volatility_20',
            'Return_5d', 'Return_10d', 'Return_20d',
            'Momentum_10', 'Momentum_20',
            'Stochastic_K', 'Stochastic_D',
            'Price_to_52W_High', 'Price_to_52W_Low',
            'sent_ticker_mean_1d', 'sent_ticker_mean_3d', 'sent_ticker_mean_5d',
            'article_count_1d', 'article_count_3d',
            'sent_momentum', 'sent_volatility_5d', 'news_volume_spike',
            'extreme_negative_news', 'extreme_positive_news'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        
        for col in available_features:
            if col not in df.columns:
                df[col] = 0
        
        df[available_features] = df[available_features].fillna(0)
        
        df[available_features] = df[available_features].replace([np.inf, -np.inf], 0)
        
        self.feature_cols = available_features
        
        return df[available_features]
    
    def create_labels(self, df: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
        df = df.copy()
        
        df['Future_Return'] = df.groupby('Ticker')['AdjClose'].shift(-horizon) / df['AdjClose'] - 1
        
        threshold = 0.05
        df['Label'] = 0
        df.loc[df['Future_Return'] > threshold, 'Label'] = 1
        df.loc[df['Future_Return'] < -threshold, 'Label'] = -1
        
        df = df.dropna(subset=['Future_Return', 'Label'])
        
        return df
    
    def train(self, df: pd.DataFrame):
        logger.info("Starting model training")
        
        df = self.create_labels(df, self.config.get('label_horizon_days', 21))
        
        X = self.prepare_features(df)
        y = df['Label']
        
        X = X[~y.isna()]
        y = y[~y.isna()]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        if self.model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        logger.info(f"Training complete - Train score: {train_score:.4f}, Test score: {test_score:.4f}")
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"Top 10 features:\n{feature_importance.head(10)}")
        
        self.save_model()
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'feature_importance': feature_importance
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            logger.warning("No trained model found, using random predictions")
            df['prediction'] = np.random.choice([-1, 0, 1], size=len(df))
            df['prediction_proba'] = np.random.uniform(0.3, 0.7, size=len(df))
            return df
        
        X = self.prepare_features(df)
        
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        
        if hasattr(self.model, 'predict_proba'):
            probas = self.model.predict_proba(X_scaled)
            max_proba = np.max(probas, axis=1)
        else:
            max_proba = np.ones(len(predictions)) * 0.5
        
        df['prediction'] = predictions
        df['prediction_proba'] = max_proba
        
        return df
    
    def save_model(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.feature_cols, "artifacts/models/feature_cols.pkl")
            logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.feature_cols = joblib.load("artifacts/models/feature_cols.pkl")
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def backtest(self, df: pd.DataFrame) -> Dict:
        df = self.create_labels(df, self.config.get('label_horizon_days', 21))
        
        predictions = self.predict(df)
        
        results = []
        
        for date in predictions['Date'].unique():
            date_preds = predictions[predictions['Date'] == date]
            
            top_n = self.config.get('top_n', 20)
            longs = date_preds[date_preds['prediction'] == 1].nlargest(top_n, 'prediction_proba')
            
            if not longs.empty:
                avg_return = longs['Future_Return'].mean()
                results.append({
                    'date': date,
                    'return': avg_return,
                    'num_positions': len(longs)
                })
        
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            results_df['cumulative_return'] = (1 + results_df['return']).cumprod()
            
            total_return = results_df['cumulative_return'].iloc[-1] - 1
            sharpe_ratio = results_df['return'].mean() / results_df['return'].std() * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(results_df['cumulative_return'])
            
            return {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'results': results_df
            }
        
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'results': pd.DataFrame()
        }
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        return drawdown.min()