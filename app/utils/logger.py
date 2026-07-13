import logging
import sys

def setup_logger():
    """Configures application-wide logging format and level."""
    logger = logging.getLogger("gapnavigator")
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if configured already
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    return logger
