#!/usr/bin/env python3
"""
Automated update pipeline for NIFTY 50 stock prediction system
Handles data updates, model retraining, and system maintenance
"""

import os
import sys
import logging
from datetime import datetime
import argparse

# Add src to path
sys.path.append('src')

from src.data import update_data_pipeline, get_nifty50_stocks
from src.train import train_all_models

# Configure logging
repo_root = os.path.dirname(os.path.abspath(__file__))  # stock-trend-predictor folder
log_file_path = os.path.join(repo_root, 'pipeline.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class PipelineUpdater:
    
    def __init__(self, data_dir="data", models_dir="models"):
                
        # Use repo root as the folder containing update_pipeline.py
        repo_root = os.path.dirname(os.path.abspath(__file__))  # stock-trend-predictor
        
        # Correct absolute paths inside repo
        self.data_dir = os.path.join(repo_root, data_dir)
        self.models_dir = os.path.join(repo_root, models_dir)

        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
    def check_nifty50_changes(self):
        """Check if NIFTY 50 composition has changed"""
        try:
            current_stocks = set(get_nifty50_stocks())
            
            # Load previous stock list if exists
            stock_list_file = os.path.join(self.data_dir, "nifty50_stocks.txt")
            
            if os.path.exists(stock_list_file):
                with open(stock_list_file, 'r') as f:
                    previous_stocks = set(line.strip() for line in f.readlines())
                
                # Check for changes
                added_stocks = current_stocks - previous_stocks
                removed_stocks = previous_stocks - current_stocks
                
                if added_stocks or removed_stocks:
                    logger.info("NIFTY 50 composition changed!")
                    if added_stocks:
                        logger.info(f"Added stocks: {added_stocks}")
                    if removed_stocks:
                        logger.info(f"Removed stocks: {removed_stocks}")
                    
                    # Update stock list file
                    with open(stock_list_file, 'w') as f:
                        for stock in sorted(current_stocks):
                            f.write(f"{stock}\n")
                    
                    return True
                else:
                    logger.info("No changes in NIFTY 50 composition")
                    return False
            else:
                # First time - save current stock list
                with open(stock_list_file, 'w') as f:
                    for stock in sorted(current_stocks):
                        f.write(f"{stock}\n")
                logger.info("Saved initial NIFTY 50 stock list")
                return True
                
        except Exception as e:
            logger.error(f"Error checking NIFTY 50 changes: {e}")
            return False
    
    def update_data(self):
        """Update stock data"""
        logger.info("Starting data update...")
        
        try:
            successful_updates = update_data_pipeline(self.data_dir)
            logger.info(f"Data update completed: {successful_updates} stocks updated")
            return True
        except Exception as e:
            logger.error(f"Data update failed: {e}")
            return False
    
    def retrain_models(self):
        """Retrain all models"""
        logger.info("Starting model retraining...")
        
        try:
            base_metrics, fine_tuned_results = train_all_models(self.data_dir, self.models_dir)
            
            logger.info("Model retraining completed successfully")
            logger.info(f"Base model MAPE: {base_metrics['MAPE']:.2f}%")
            
            for stock, metrics in fine_tuned_results.items():
                logger.info(f"Fine-tuned {stock} MAPE: {metrics['MAPE']:.2f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return False
    
    def run_full_update(self, force_retrain=False):
        """Run complete update pipeline"""
        logger.info("="*60)
        logger.info("STARTING AUTOMATED UPDATE PIPELINE")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        # Step 1: Check for NIFTY 50 changes
        nifty_changed = self.check_nifty50_changes()
        
        # Step 2: Update data
        data_updated = self.update_data()
        
        if not data_updated:
            logger.error("Data update failed. Aborting pipeline.")
            return False
        
        # Step 3: Retrain models if needed
        should_retrain = force_retrain or nifty_changed
        
        if should_retrain:
            logger.info("Retraining models due to changes...")
            model_updated = self.retrain_models()
            
            if not model_updated:
                logger.error("Model retraining failed.")
                return False
        else:
            logger.info("No model retraining required")
        
        # Step 4: Cleanup
        self.cleanup_old_logs()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("="*60)
        logger.info("UPDATE PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Duration: {duration}")
        logger.info("="*60)
        
        return True
    
    def cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            log_files = [f for f in os.listdir('.') if f.endswith('.log')]
            
            # Keep only last 10 log files
            if len(log_files) > 10:
                log_files.sort(key=lambda x: os.path.getmtime(x))
                for old_log in log_files[:-10]:
                    os.remove(old_log)
                    logger.info(f"Removed old log file: {old_log}")
                    
        except Exception as e:
            logger.warning(f"Log cleanup failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="NIFTY 50 Stock Prediction Update Pipeline")
    parser.add_argument('--data-only', action='store_true', help='Update data only, no model retraining')
    parser.add_argument('--force-retrain', action='store_true', help='Force model retraining even if no changes')
    parser.add_argument('--check-nifty', action='store_true', help='Only check for NIFTY 50 changes')
    parser.add_argument('--data-dir', default='data', help='Data directory path')
    parser.add_argument('--models-dir', default='models', help='Models directory path')
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = PipelineUpdater(args.data_dir, args.models_dir)
    
    try:
        if args.check_nifty:
            # Only check for NIFTY 50 changes
            changed = updater.check_nifty50_changes()
            print(f"NIFTY 50 composition changed: {changed}")
            
        elif args.data_only:
            # Update data only
            logger.info("Running data-only update...")
            success = updater.update_data()
            sys.exit(0 if success else 1)
            
        else:
            # Full pipeline
            success = updater.run_full_update(force_retrain=args.force_retrain)
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()