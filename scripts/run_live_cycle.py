#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from src.jobs.scheduler import TradingScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting live trading cycle")
    
    scheduler = TradingScheduler()
    
    logger.info("Updating data...")
    scheduler.update_data()
    
    logger.info("Building features...")
    scheduler.build_features()
    
    logger.info("Generating signals...")
    scheduler.generate_signals()
    
    logger.info("Executing trades...")
    scheduler.execute_trades()
    
    logger.info("Reconciling positions...")
    scheduler.reconcile_positions()
    
    logger.info("Live cycle complete")

if __name__ == "__main__":
    main()