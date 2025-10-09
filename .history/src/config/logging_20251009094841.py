# Logging Configuration
import logging
import os
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log"):
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Set specific loggers
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("fastapi").setLevel(logging.INFO)
    logging.getLogger("streamlit").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

# Default logger
logger = setup_logging()
