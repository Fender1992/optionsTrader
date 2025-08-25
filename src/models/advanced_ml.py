import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier, VotingRegressor
import lightgbm as lgb
import xgboost as xgb
from typing import Dict, List, Tuple, Optional, Union
import joblib
import os
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedMLSignalGenerator:
    """
    Advanced ML signal generator using ensemble of XGBoost and LightGBM
    with sophisticated feature engineering and walk-forward optimization.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_type = config.get('model_type', 'ensemble')  # 'xgboost', 'lightgbm', or 'ensemble'
        self.prediction_horizon = config.get('prediction_horizon', 5)  # Days ahead
        self.min_confidence = config.get('min_confidence', 0.6)
        
        # Models
        self.models = {}
        self.scalers = {}
        self.feature_cols = None
        
        # Performance tracking
        self.performance_history = []
        self.feature_importance = {}
        
        # Model paths
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir, exist_ok=True)
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and engineer features for ML models.
        Includes all technical indicators plus engineered features.
        """
        df = df.copy()
        
        # Core feature columns (from advanced_technical.py)
        base_features = [
            # Returns
            'Return_1d', 'Return_5d', 'Return_10d', 'Return_21d',
            'LogReturn_1d', 'LogReturn_5d', 'LogReturn_21d',
            
            # Momentum
            'RSI_7', 'RSI_14', 'RSI_21',
            'Momentum_10', 'Momentum_20', 'Momentum_40',
            'ROC_10', 'ROC_20', 'ROC_30',
            'TSI', 'PPO', 'TRIX',
            
            # Volatility
            'Volatility_5', 'Volatility_10', 'Volatility_20', 'Volatility_60',
            'ATR_14', 'ATR_20', 'ATR_Ratio_14', 'ATR_Ratio_20',
            'BB_Width_20', 'BB_Position_20',
            'GK_Volatility', 'Parkinson_Vol',
            
            # Volume
            'Volume_Ratio_5', 'Volume_Ratio_10', 'Volume_Ratio_20',
            'OBV', 'CMF', 'MFI', 'ADL', 'Price_to_VWAP',
            
            # Trend
            'ADX', 'DI_Plus', 'DI_Minus',
            'Aroon_Up', 'Aroon_Down', 'Aroon_Oscillator',
            'MACD', 'MACD_Signal', 'MACD_Histogram',
            
            # Statistical
            'Skew_20', 'Kurtosis_20', 'Z_Score_20',
            'Sharpe_20', 'Sortino_20',
            'Max_Drawdown_20', 'Max_Drawdown_50',
            
            # Market Regime
            'Market_Phase', 'Vol_Regime', 'Trend_Strength',
            'Momentum_Quality', 'Trend_Consistency',
            
            # Options-specific
            'Vol_Term_Structure', 'Vol_of_Vol', 'Vol_Spike',
            'Mean_Reversion_Z_20', 'Mean_Reversion_Z_50',
            'Price_Acceleration', 'Theta_Favorable'
        ]
        
        # News sentiment features (if available)
        sentiment_features = [
            'sent_ticker_mean_1d', 'sent_ticker_mean_3d', 'sent_ticker_mean_5d',
            'article_count_1d', 'article_count_3d',
            'sent_momentum', 'sent_volatility_5d',
            'extreme_negative_news', 'extreme_positive_news'
        ]
        
        # Combine all available features
        available_features = []
        for col in base_features + sentiment_features:
            if col in df.columns:
                available_features.append(col)
        
        # Engineer additional interaction features
        df = self._engineer_interaction_features(df, available_features)
        
        # Add time-based features
        df = self._add_time_features(df)
        
        # Add cross-sectional features (if multiple tickers)
        df = self._add_cross_sectional_features(df)
        
        self.feature_cols = available_features + list(df.columns[df.columns.str.startswith('eng_')])
        
        return df
    
    def _engineer_interaction_features(self, df: pd.DataFrame, base_features: List[str]) -> pd.DataFrame:
        """Engineer interaction and polynomial features."""
        
        # Key interaction features for options trading
        if 'RSI_14' in df.columns and 'Volatility_20' in df.columns:
            df['eng_rsi_vol_interaction'] = df['RSI_14'] * df['Volatility_20']
        
        if 'Momentum_20' in df.columns and 'Volume_Ratio_20' in df.columns:
            df['eng_momentum_volume'] = df['Momentum_20'] * df['Volume_Ratio_20']
        
        if 'BB_Position_20' in df.columns and 'RSI_14' in df.columns:
            df['eng_bb_rsi_signal'] = df['BB_Position_20'] * df['RSI_14'] / 100
        
        if 'Vol_Term_Structure' in df.columns and 'Mean_Reversion_Z_20' in df.columns:
            df['eng_vol_mean_rev'] = df['Vol_Term_Structure'] * df['Mean_Reversion_Z_20']
        
        # Ratios
        if 'Volatility_5' in df.columns and 'Volatility_20' in df.columns:
            df['eng_vol_ratio_5_20'] = df['Volatility_5'] / (df['Volatility_20'] + 0.0001)
        
        if 'ATR_14' in df.columns and 'Volatility_20' in df.columns:
            df['eng_atr_vol_divergence'] = df['ATR_14'] / (df['Volatility_20'] + 0.0001)
        
        # Composite signals
        if all(col in df.columns for col in ['RSI_14', 'MACD_Histogram', 'Momentum_20']):
            df['eng_composite_momentum'] = (
                df['RSI_14'] / 100 + 
                np.sign(df['MACD_Histogram']) + 
                np.sign(df['Momentum_20'])
            ) / 3
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features relevant for options."""
        
        if 'Date' in df.columns:
            df['eng_day_of_week'] = pd.to_datetime(df['Date']).dt.dayofweek
            df['eng_day_of_month'] = pd.to_datetime(df['Date']).dt.day
            df['eng_month'] = pd.to_datetime(df['Date']).dt.month
            df['eng_quarter'] = pd.to_datetime(df['Date']).dt.quarter
            
            # Options expiration related (third Friday)
            df['eng_days_to_monthly_expiry'] = df.apply(
                lambda x: self._days_to_third_friday(pd.to_datetime(x['Date'])), axis=1
            )
            df['eng_is_expiry_week'] = (df['eng_days_to_monthly_expiry'] <= 5).astype(int)
        
        return df
    
    def _days_to_third_friday(self, date: pd.Timestamp) -> int:
        """Calculate days to the third Friday of the month."""
        # Find third Friday
        year, month = date.year, date.month
        
        # First day of month
        first_day = pd.Timestamp(year, month, 1)
        
        # Find first Friday
        days_until_friday = (4 - first_day.dayofweek) % 7
        first_friday = first_day + pd.Timedelta(days=days_until_friday)
        
        # Third Friday is 14 days after first Friday
        third_friday = first_friday + pd.Timedelta(days=14)
        
        # Days until third Friday
        days_to_expiry = (third_friday - date).days
        
        # If past third Friday, calculate for next month
        if days_to_expiry < 0:
            if month == 12:
                next_month = pd.Timestamp(year + 1, 1, 1)
            else:
                next_month = pd.Timestamp(year, month + 1, 1)
            
            days_until_friday = (4 - next_month.dayofweek) % 7
            first_friday = next_month + pd.Timedelta(days=days_until_friday)
            third_friday = first_friday + pd.Timedelta(days=14)
            days_to_expiry = (third_friday - date).days
        
        return days_to_expiry
    
    def _add_cross_sectional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-sectional features comparing to market/sector."""
        
        if 'Ticker' in df.columns and df['Ticker'].nunique() > 1:
            # Calculate market averages
            for col in ['Return_5d', 'Volatility_20', 'RSI_14', 'Volume_Ratio_20']:
                if col in df.columns:
                    market_avg = df.groupby('Date')[col].transform('mean')
                    df[f'eng_{col}_vs_market'] = df[col] - market_avg
                    
                    # Percentile rank
                    df[f'eng_{col}_rank'] = df.groupby('Date')[col].rank(pct=True)
        
        return df
    
    def create_labels(self, df: pd.DataFrame, target_type: str = 'classification') -> pd.DataFrame:
        """
        Create labels for ML models.
        
        Args:
            target_type: 'classification' (direction) or 'regression' (returns)
        """
        df = df.copy()
        
        # Calculate forward returns
        df['forward_return'] = df.groupby('Ticker')['AdjClose'].shift(-self.prediction_horizon) / df['AdjClose'] - 1
        
        if target_type == 'classification':
            # Multi-class: strong bearish, bearish, neutral, bullish, strong bullish
            conditions = [
                df['forward_return'] < -0.03,  # Strong bearish
                (df['forward_return'] >= -0.03) & (df['forward_return'] < -0.01),  # Bearish
                (df['forward_return'] >= -0.01) & (df['forward_return'] <= 0.01),  # Neutral
                (df['forward_return'] > 0.01) & (df['forward_return'] <= 0.03),  # Bullish
                df['forward_return'] > 0.03  # Strong bullish
            ]
            choices = [0, 1, 2, 3, 4]
            df['label'] = np.select(conditions, choices, default=2)
            
            # Also create binary label for simplified trading
            df['label_binary'] = (df['forward_return'] > 0.01).astype(int)
        else:
            df['label'] = df['forward_return']
        
        return df
    
    def train_models(self, df: pd.DataFrame, target_type: str = 'classification'):
        """Train ensemble of XGBoost and LightGBM models."""
        
        logger.info("Starting model training...")
        
        # Prepare features and labels
        df = self.prepare_features(df)
        df = self.create_labels(df, target_type)
        
        # Remove NaN values
        df = df.dropna(subset=self.feature_cols + ['label'])
        
        X = df[self.feature_cols]
        y = df['label']
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train XGBoost
        if self.model_type in ['xgboost', 'ensemble']:
            xgb_model = self._train_xgboost(X_scaled, y, target_type, tscv)
            self.models['xgboost'] = xgb_model
        
        # Train LightGBM
        if self.model_type in ['lightgbm', 'ensemble']:
            lgb_model = self._train_lightgbm(X_scaled, y, target_type, tscv)
            self.models['lightgbm'] = lgb_model
        
        # Create ensemble if needed
        if self.model_type == 'ensemble':
            if target_type == 'classification':
                self.models['ensemble'] = VotingClassifier(
                    estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
                    voting='soft'
                )
            else:
                self.models['ensemble'] = VotingRegressor(
                    estimators=[('xgb', xgb_model), ('lgb', lgb_model)]
                )
            
            self.models['ensemble'].fit(X_scaled, y)
        
        # Save scaler
        self.scalers['main'] = scaler
        
        # Calculate feature importance
        self._calculate_feature_importance(X)
        
        # Save models
        self.save_models()
        
        logger.info("Model training completed")
        
        return self
    
    def _train_xgboost(self, X: np.ndarray, y: np.ndarray, 
                       target_type: str, cv_splitter) -> xgb.XGBClassifier:
        """Train XGBoost model with hyperparameter tuning."""
        
        if target_type == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
                eval_metric='rmse'
            )
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9]
        }
        
        # Random search with time series CV
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=10, cv=cv_splitter,
            scoring='f1_weighted' if target_type == 'classification' else 'neg_mean_squared_error',
            n_jobs=-1, random_state=42
        )
        
        random_search.fit(X, y)
        
        logger.info(f"XGBoost best params: {random_search.best_params_}")
        
        return random_search.best_estimator_
    
    def _train_lightgbm(self, X: np.ndarray, y: np.ndarray,
                        target_type: str, cv_splitter) -> lgb.LGBMClassifier:
        """Train LightGBM model with hyperparameter tuning."""
        
        if target_type == 'classification':
            model = lgb.LGBMClassifier(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=200,
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'num_leaves': [20, 31, 40],
            'learning_rate': [0.01, 0.05, 0.1],
            'feature_fraction': [0.7, 0.8, 0.9]
        }
        
        # Random search with time series CV
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=10, cv=cv_splitter,
            scoring='f1_weighted' if target_type == 'classification' else 'neg_mean_squared_error',
            n_jobs=-1, random_state=42
        )
        
        random_search.fit(X, y)
        
        logger.info(f"LightGBM best params: {random_search.best_params_}")
        
        return random_search.best_estimator_
    
    def predict(self, df: pd.DataFrame, return_proba: bool = True) -> pd.DataFrame:
        """Generate predictions with confidence scores."""
        
        if not self.models:
            logger.error("No trained models found")
            return df
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Get feature columns and handle missing
        X = df[self.feature_cols].fillna(0)
        
        # Scale features
        if 'main' in self.scalers:
            X_scaled = self.scalers['main'].transform(X)
        else:
            X_scaled = X.values
        
        # Get predictions from the active model
        if self.model_type == 'ensemble' and 'ensemble' in self.models:
            model = self.models['ensemble']
        elif 'xgboost' in self.models:
            model = self.models['xgboost']
        elif 'lightgbm' in self.models:
            model = self.models['lightgbm']
        else:
            logger.error("No valid model found")
            return df
        
        # Make predictions
        if hasattr(model, 'predict_proba') and return_proba:
            probas = model.predict_proba(X_scaled)
            df['prediction'] = np.argmax(probas, axis=1)
            df['confidence'] = np.max(probas, axis=1)
            
            # Add probability for each class
            for i in range(probas.shape[1]):
                df[f'prob_class_{i}'] = probas[:, i]
            
            # Trading signals
            df['signal_strength'] = df['confidence']
            df['signal_direction'] = df['prediction'].map({
                0: 'strong_bearish',
                1: 'bearish',
                2: 'neutral',
                3: 'bullish',
                4: 'strong_bullish'
            })
        else:
            predictions = model.predict(X_scaled)
            df['prediction'] = predictions
            df['confidence'] = 0.5  # Default confidence
            df['signal_strength'] = abs(predictions)
            df['signal_direction'] = np.where(predictions > 0.01, 'bullish',
                                             np.where(predictions < -0.01, 'bearish', 'neutral'))
        
        # Add meta information
        df['model_type'] = self.model_type
        df['prediction_horizon'] = self.prediction_horizon
        df['prediction_timestamp'] = datetime.now()
        
        return df
    
    def _calculate_feature_importance(self, X: pd.DataFrame):
        """Calculate and store feature importance."""
        
        importance_dict = {}
        
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            importance_dict['xgboost'] = dict(zip(self.feature_cols, xgb_importance))
        
        if 'lightgbm' in self.models:
            lgb_importance = self.models['lightgbm'].feature_importances_
            importance_dict['lightgbm'] = dict(zip(self.feature_cols, lgb_importance))
        
        # Average importance if ensemble
        if len(importance_dict) > 1:
            avg_importance = {}
            for feature in self.feature_cols:
                avg_importance[feature] = np.mean([
                    imp.get(feature, 0) for imp in importance_dict.values()
                ])
            importance_dict['ensemble'] = avg_importance
        
        self.feature_importance = importance_dict
        
        # Log top features
        if 'ensemble' in importance_dict:
            top_features = sorted(importance_dict['ensemble'].items(), 
                                key=lambda x: x[1], reverse=True)[:20]
            logger.info("Top 20 features:")
            for feat, imp in top_features:
                logger.info(f"  {feat}: {imp:.4f}")
    
    def backtest_predictions(self, df: pd.DataFrame, 
                            start_date: str = None,
                            end_date: str = None) -> Dict:
        """Backtest model predictions with walk-forward analysis."""
        
        results = {
            'dates': [],
            'predictions': [],
            'actuals': [],
            'returns': [],
            'cumulative_return': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Filter by date range if provided
        if start_date:
            df = df[df['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['Date'] <= pd.to_datetime(end_date)]
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Walk-forward analysis
        window_size = 252  # 1 year training window
        step_size = 21  # Retrain monthly
        
        for i in range(window_size, len(df), step_size):
            train_data = df.iloc[i-window_size:i]
            test_data = df.iloc[i:min(i+step_size, len(df))]
            
            if len(test_data) == 0:
                break
            
            # Train on window
            self.train_models(train_data)
            
            # Predict on test
            predictions = self.predict(test_data)
            
            # Calculate metrics
            if 'label' in test_data.columns:
                y_true = test_data['label'].values
                y_pred = predictions['prediction'].values
                
                results['dates'].extend(test_data['Date'].values)
                results['predictions'].extend(y_pred)
                results['actuals'].extend(y_true)
                
                # Performance metrics
                results['accuracy'].append(accuracy_score(y_true, y_pred))
                results['precision'].append(precision_score(y_true, y_pred, average='weighted'))
                results['recall'].append(recall_score(y_true, y_pred, average='weighted'))
                results['f1'].append(f1_score(y_true, y_pred, average='weighted'))
        
        return results
    
    def save_models(self):
        """Save trained models and scalers."""
        
        for name, model in self.models.items():
            model_path = os.path.join(self.model_dir, f"{name}_model.pkl")
            joblib.dump(model, model_path)
            logger.info(f"Saved {name} model to {model_path}")
        
        for name, scaler in self.scalers.items():
            scaler_path = os.path.join(self.model_dir, f"{name}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
        
        # Save feature columns
        feature_path = os.path.join(self.model_dir, "feature_cols.pkl")
        joblib.dump(self.feature_cols, feature_path)
        
        # Save feature importance
        importance_path = os.path.join(self.model_dir, "feature_importance.pkl")
        joblib.dump(self.feature_importance, importance_path)
    
    def load_models(self):
        """Load saved models and scalers."""
        
        # Load models
        for model_type in ['xgboost', 'lightgbm', 'ensemble']:
            model_path = os.path.join(self.model_dir, f"{model_type}_model.pkl")
            if os.path.exists(model_path):
                self.models[model_type] = joblib.load(model_path)
                logger.info(f"Loaded {model_type} model")
        
        # Load scalers
        scaler_path = os.path.join(self.model_dir, "main_scaler.pkl")
        if os.path.exists(scaler_path):
            self.scalers['main'] = joblib.load(scaler_path)
        
        # Load feature columns
        feature_path = os.path.join(self.model_dir, "feature_cols.pkl")
        if os.path.exists(feature_path):
            self.feature_cols = joblib.load(feature_path)
        
        # Load feature importance
        importance_path = os.path.join(self.model_dir, "feature_importance.pkl")
        if os.path.exists(importance_path):
            self.feature_importance = joblib.load(importance_path)