import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from typing import Union, List, Dict, Callable, Any, Optional, Tuple
from datetime import datetime
from functools import wraps
import hashlib
import pickle

#Configure Logging:
logging.basicConfig(
    level = logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(self, config_path: str) -> None:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        self.config = json.load(f)
    logger.info(f"Configuration loaded from {config_path}")
    
def save_config(self, config_path: str) -> None:
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(self.config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")
    
def get(self, key: str, default: Any = None) -> Any:
    """Get configuration value."""
    return self.config.get(key, default)
    
def set(self, key: str, value: Any) -> None:
    """Set configuration value."""
    self.config[key] = value


def pipeline_step(step_name: str, log_output: bool = True):
    """
    Decorator for pipeline steps with logging and error handling.
    
    Parameters
    ----------
    step_name : str
        Name of the pipeline step
    log_output : bool, default True
        Whether to log output shape/type
    
    Returns
    -------
    Callable
        Decorated function
    
    Examples
    --------
    @pipeline_step("data_loading")
        def load_data(path):
            return pd.read_csv(path)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"Starting pipeline step: {step_name}")
            start_time = datetime.now()
            
            try:
                result = func(*args, **kwargs)
                elapsed = (datetime.now() - start_time).total_seconds()
                
                if log_output:
                    if isinstance(result, pd.DataFrame):
                        logger.info(
                            f"Step '{step_name}' completed in {elapsed:.2f}s. "
                            f"Output shape: {result.shape}"
                        )
                    else:
                        logger.info(
                            f"Step '{step_name}' completed in {elapsed:.2f}s. "
                            f"Output type: {type(result).__name__}"
                        )
                else:
                    logger.info(f"Step '{step_name}' completed in {elapsed:.2f}s")
                
                return result
            
            except Exception as e:
                logger.error(f"Error in step '{step_name}': {str(e)}")
                raise
        
            return wrapper
        return decorator