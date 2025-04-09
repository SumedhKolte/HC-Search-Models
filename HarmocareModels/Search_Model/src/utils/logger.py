import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime, timezone

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Only add handlers if they haven't been added yet
    if not logger.handlers:
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # Create and setup file handler
        log_dir = Path(__file__).resolve().parent.parent.parent / 'monitoring' / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d')
        log_file = log_dir / f'{name}_{timestamp}.log'
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        
        # Create and setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# Create default logger instance
default_logger = setup_logger('medical_search')

# Export setup_logger function and default logger
__all__ = ['setup_logger', 'default_logger']