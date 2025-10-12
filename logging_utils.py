import logging
import sys
from datetime import datetime

def setup_logging(log_file=None):
    """Setup logging để ghi vào cả terminal và file"""
    if log_file is None:
        log_file = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Tạo formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    # Console handler (terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file

def log_gpu_memory():
    """Log GPU memory usage"""
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3   # GB
            logging.info(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
