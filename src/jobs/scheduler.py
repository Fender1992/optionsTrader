import os
import json
import logging
from datetime import datetime, time, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import Dict, List, Optional
import pandas as pd
import yaml

from src.fetchers.alpha_vantage import AlphaVantageFetcher
from src.features.news import NewsFeatureBuilder
from src.features.technical import TechnicalFeatureBuilder
from src.options.data import OptionsDataInterface
from src.options.strategy import OptionsStrategyEngine
from src.execution.broker import TradierBroker
from src.execution.sizer import PortfolioSizer
from src.models.ml_model import MLModel
from src.alerts.notifier import EquityMilestoneTracker

logger = logging.getLogger(__name__)

class TradingScheduler:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scheduler = BackgroundScheduler()
        
        self.alpha_fetcher = AlphaVantageFetcher()
        self.news_builder = NewsFeatureBuilder(self.config)
        self.technical_builder = TechnicalFeatureBuilder(self.config)
        self.options_data = OptionsDataInterface(broker="tradier")
        self.strategy_engine = OptionsStrategyEngine(self.config, self.options_data)
        self.broker = TradierBroker(self.config)
        self.sizer = PortfolioSizer(self.config)
        self.ml_model = MLModel(self.config)
        self.milestone_tracker = EquityMilestoneTracker(self.config)
        
        self.state_file = "artifacts/live/trading_state.json"
        self.pnl_file = "artifacts/live/pnl_history.csv"
        self.log_file = "artifacts/live/trading_log.json"
        
        self._ensure_directories()
        self.trading_state = self._load_state()
        
    def _ensure_directories(self):
        os.makedirs("artifacts/live", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs(".cache", exist_ok=True)
    
    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state: {e}")
        
        initial_capital = float(os.getenv('CAPITAL_DOLLARS', 10000))
        
        # Initialize milestone tracker baseline
        self.milestone_tracker.initialize_baseline(initial_capital)
        
        return {
            'last_update': None,
            'positions': [],
            'pending_orders': [],
            'realized_pnl': 0,
            'capital': initial_capital,
            'initial_capital': initial_capital
        }
    
    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.trading_state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _log_action(self, action: str, details: Dict):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            if len(logs) > 1000:
                logs = logs[-500:]
            
            with open(self.log_file, 'w') as f:
                json.dump(logs, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to write log: {e}")
    
    def update_data(self):
        logger.info("Starting data update job")
        self._log_action("data_update", {"status": "started"})
        
        try:
            symbols = self._get_universe()
            
            for symbol in symbols:
                try:
                    logger.info(f"Fetching data for {symbol}")
                    
                    price_data = self.alpha_fetcher.fetch_daily_adjusted(
                        symbol,
                        start_date=self.config.get('start_date'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )
                    
                    logger.info(f"Fetched {len(price_data)} price records for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Failed to fetch price data for {symbol}: {e}")
            
            news_data = self.alpha_fetcher.fetch_news_sentiment(
                symbols,
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
            
            logger.info(f"Fetched {len(news_data)} news articles")
            
            self._log_action("data_update", {
                "status": "completed",
                "symbols": len(symbols),
                "news_articles": len(news_data)
            })
            
        except Exception as e:
            logger.error(f"Data update failed: {e}")
            self._log_action("data_update", {"status": "failed", "error": str(e)})
    
    def build_features(self):
        logger.info("Building features")
        self._log_action("feature_build", {"status": "started"})
        
        try:
            symbols = self._get_universe()
            
            all_price_data = []
            for symbol in symbols:
                cache_file = f".cache/{symbol}_daily.parquet"
                if os.path.exists(cache_file):
                    df = pd.read_parquet(cache_file)
                    all_price_data.append(df)
            
            if all_price_data:
                price_df = pd.concat(all_price_data, ignore_index=True)
                
                technical_features = self.technical_builder.build_features(price_df)
                
                news_cache_files = [f".cache/{s}_news.parquet" for s in symbols]
                news_dfs = []
                for file in news_cache_files:
                    if os.path.exists(file):
                        news_dfs.append(pd.read_parquet(file))
                
                if news_dfs:
                    news_df = pd.concat(news_dfs, ignore_index=True)
                    news_features = self.news_builder.build_news_features(news_df, price_df)
                else:
                    news_features = pd.DataFrame()
                
                if not technical_features.empty and not news_features.empty:
                    features = pd.merge(
                        technical_features,
                        news_features,
                        on=['Date', 'ticker'],
                        how='left'
                    )
                else:
                    features = technical_features
                
                features.to_parquet("data/features.parquet", index=False)
                
                self._log_action("feature_build", {
                    "status": "completed",
                    "rows": len(features),
                    "columns": len(features.columns)
                })
                
                logger.info(f"Features built: {len(features)} rows, {len(features.columns)} columns")
                
        except Exception as e:
            logger.error(f"Feature building failed: {e}")
            self._log_action("feature_build", {"status": "failed", "error": str(e)})
    
    def generate_signals(self):
        logger.info("Generating trading signals")
        self._log_action("signal_generation", {"status": "started"})
        
        try:
            if not os.path.exists("data/features.parquet"):
                logger.warning("Features not found, building first")
                self.build_features()
            
            features = pd.read_parquet("data/features.parquet")
            
            predictions = self.ml_model.predict(features)
            
            latest_date = features['Date'].max()
            latest_predictions = predictions[predictions['Date'] == latest_date]
            
            ranked_symbols = []
            for _, row in latest_predictions.iterrows():
                ranked_symbols.append({
                    'symbol': row['Ticker'],
                    'signal_strength': row.get('prediction_proba', 0.5),
                    'prediction': 'bullish' if row.get('prediction', 0) > 0 else 'bearish'
                })
            
            ranked_symbols.sort(key=lambda x: abs(x['signal_strength'] - 0.5), reverse=True)
            
            current_positions = self.broker.get_positions()
            
            news_features = features[features['Date'] == latest_date][['ticker', 'extreme_negative_news', 'sent_ticker_mean_1d']]
            
            signals = self.strategy_engine.generate_signals(
                ranked_symbols[:self.config['top_n']],
                news_features,
                current_positions
            )
            
            self.trading_state['pending_signals'] = signals
            self._save_state()
            
            self._log_action("signal_generation", {
                "status": "completed",
                "ranked_symbols": len(ranked_symbols),
                "signals": len(signals)
            })
            
            logger.info(f"Generated {len(signals)} trading signals")
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            self._log_action("signal_generation", {"status": "failed", "error": str(e)})
    
    def execute_trades(self):
        logger.info("Executing trades")
        self._log_action("trade_execution", {"status": "started"})
        
        try:
            signals = self.trading_state.get('pending_signals', [])
            
            if not signals:
                logger.info("No signals to execute")
                return
            
            account_info = self.broker.get_account_info()
            current_positions = self.broker.get_positions()
            
            capital = self.trading_state.get('capital', 10000)
            
            # Pass account_info for compounding
            sized_orders = self.sizer.size_positions(
                capital,
                signals,
                current_positions,
                account_info=account_info
            )
            
            executed_orders = []
            for order in sized_orders:
                try:
                    result = self.broker.place_order(order)
                    executed_orders.append({
                        'order': order,
                        'result': result
                    })
                    
                    logger.info(f"Order placed: {result}")
                    
                except Exception as e:
                    logger.error(f"Failed to place order for {order['symbol']}: {e}")
            
            self.trading_state['last_execution'] = datetime.now().isoformat()
            self.trading_state['executed_orders'] = executed_orders
            self.trading_state['pending_signals'] = []
            self._save_state()
            
            self._log_action("trade_execution", {
                "status": "completed",
                "orders_executed": len(executed_orders)
            })
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            self._log_action("trade_execution", {"status": "failed", "error": str(e)})
    
    def reconcile_positions(self):
        logger.info("Reconciling positions and PnL")
        
        try:
            positions = self.broker.get_positions()
            account_info = self.broker.get_account_info()
            
            total_equity = account_info.get('total_equity', 0)
            
            pnl_data = {
                'timestamp': datetime.now(),
                'total_equity': total_equity,
                'cash': account_info.get('cash', 0),
                'num_positions': len(positions),
                'unrealized_pnl': sum(p.get('unrealized_pnl', 0) for p in positions),
                'realized_pnl': sum(p.get('realized_pnl', 0) for p in positions)
            }
            
            if os.path.exists(self.pnl_file):
                pnl_history = pd.read_csv(self.pnl_file)
            else:
                pnl_history = pd.DataFrame()
            
            new_row = pd.DataFrame([pnl_data])
            pnl_history = pd.concat([pnl_history, new_row], ignore_index=True)
            pnl_history.to_csv(self.pnl_file, index=False)
            
            self.trading_state['positions'] = positions
            self.trading_state['account_info'] = account_info
            self.trading_state['last_reconcile'] = datetime.now().isoformat()
            
            # Update capital for compounding (reinvest all gains)
            self.trading_state['capital'] = total_equity
            
            self._save_state()
            
            logger.info(f"Reconciled {len(positions)} positions, equity: ${total_equity:,.2f}")
            
            # Check for equity milestones
            if total_equity > 0:
                milestone = self.milestone_tracker.check_milestone(total_equity)
                if milestone:
                    self._log_action("equity_milestone", milestone)
            
        except Exception as e:
            logger.error(f"Position reconciliation failed: {e}")
    
    def _get_universe(self) -> List[str]:
        universe_file = self.config.get('universe_file', 'data/symbols.csv')
        
        if os.path.exists(universe_file):
            df = pd.read_csv(universe_file)
            return df['symbol'].tolist()
        else:
            default_symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']
            
            pd.DataFrame({'symbol': default_symbols}).to_csv(universe_file, index=False)
            
            return default_symbols
    
    def start(self):
        self.scheduler.add_job(
            self.update_data,
            CronTrigger(hour=8, minute=30),
            id='update_data',
            name='Update market data',
            replace_existing=True
        )
        
        self.scheduler.add_job(
            self.build_features,
            CronTrigger(hour=9, minute=0),
            id='build_features',
            name='Build features',
            replace_existing=True
        )
        
        self.scheduler.add_job(
            self.generate_signals,
            CronTrigger(hour=9, minute=20),
            id='generate_signals',
            name='Generate trading signals',
            replace_existing=True
        )
        
        rebalance_day = {'W': 'mon', 'M': 1}.get(self.config.get('rebalance_freq', 'W'), 'mon')
        
        if isinstance(rebalance_day, str):
            self.scheduler.add_job(
                self.execute_trades,
                CronTrigger(day_of_week=rebalance_day, hour=9, minute=35),
                id='execute_trades',
                name='Execute trades',
                replace_existing=True
            )
        else:
            self.scheduler.add_job(
                self.execute_trades,
                CronTrigger(day=rebalance_day, hour=9, minute=35),
                id='execute_trades',
                name='Execute trades',
                replace_existing=True
            )
        
        self.scheduler.add_job(
            self.reconcile_positions,
            CronTrigger(hour=16, minute=0),
            id='reconcile_positions',
            name='Reconcile positions',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("Trading scheduler started")
    
    def stop(self):
        self.scheduler.shutdown()
        logger.info("Trading scheduler stopped")