import schedule
import time
import threading
from datetime import datetime
import logging

from backend.models.trainer import ModelTrainer

logger = logging.getLogger(__name__)

class ModelTrainingScheduler:
    def __init__(self):
        self.trainer = ModelTrainer()
        self.is_running = False
        self.last_training = None
        self.current_model_version = None
    
    def train_job(self):

        logger.info("🔄 Scheduled training started")
        try:
            result = self.trainer.train_new_model(training_days=30)
            self.last_training = datetime.now()
            self.current_model_version = result['version']
            logger.info(f"✅ Scheduled training complete. Version: {result['version']}")
        except Exception as e:
            logger.error(f"❌ Scheduled training failed: {e}")
    
    def start(self, schedule_time: str = "02:00", day: str = "monday"):
        if self.is_running:
            logger.warning("Scheduler already running")
            return
        
        logger.info("Running initial training on startup...")
        threading.Thread(target=self.train_job, daemon=True).start()
        
        if day.lower() == "daily":
            schedule.every().day.at(schedule_time).do(self.train_job)
        else:
            getattr(schedule.every(), day.lower()).at(schedule_time).do(self.train_job)
        
        logger.info(f"🚀 Scheduler started: weekly {day} at {schedule_time}")
        
        def run_scheduler():
            self.is_running = True
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)
        
        thread = threading.Thread(target=run_scheduler, daemon=True)
        thread.start()
    
    def stop(self):
        
        self.is_running = False
        logger.info("Scheduler stopped")
    
    def get_status(self) -> dict:
        
        return {
            "is_running": self.is_running,
            "last_training": self.last_training.isoformat() if self.last_training else None,
            "current_model_version": self.current_model_version
        }