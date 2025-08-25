#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yaml
import logging
from src.models.ml_model import MLModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    if not os.path.exists('data/features.parquet'):
        logger.error("Features not found. Run build_feature_store.py first")
        return
    
    features = pd.read_parquet('data/features.parquet')
    logger.info(f"Loaded features: {len(features)} rows")
    
    ml_model = MLModel(config)
    
    logger.info("Training model")
    train_results = ml_model.train(features)
    
    logger.info(f"Training results:")
    logger.info(f"  Train score: {train_results['train_score']:.4f}")
    logger.info(f"  Test score: {train_results['test_score']:.4f}")
    
    logger.info("\nTop 10 important features:")
    for _, row in train_results['feature_importance'].head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    logger.info("\nRunning backtest")
    backtest_results = ml_model.backtest(features)
    
    logger.info(f"\nBacktest Results:")
    logger.info(f"  Total Return: {backtest_results['total_return']:.2%}")
    logger.info(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    
    if not backtest_results['results'].empty:
        results_df = backtest_results['results']
        
        os.makedirs('artifacts', exist_ok=True)
        results_df.to_csv('artifacts/backtest_results.csv', index=False)
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        axes[0].plot(pd.to_datetime(results_df['date']), results_df['cumulative_return'])
        axes[0].set_title('Cumulative Returns')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Cumulative Return')
        axes[0].grid(True)
        
        axes[1].bar(pd.to_datetime(results_df['date']), results_df['return'])
        axes[1].set_title('Daily Returns')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Return')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('artifacts/backtest_chart.png')
        logger.info("Backtest chart saved to artifacts/backtest_chart.png")

if __name__ == "__main__":
    main()