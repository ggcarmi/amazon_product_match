"""
Logging configuration for the product matching service
"""

import os
import logging
from pathlib import Path

def setup_logging(config):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure log file path
    log_file = log_dir / config['log_file']
    
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(str(log_file)),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("ProductMatcher")