"""
Crime Research Data Pipeline - Class Definition For Refactored Data Utilities

This module defines the newDataStorageUtils class, a refactored implementation
of the dataStorageUtils class from the earlier crime research data pipeline core
classes implementation.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

import os
import pandas as pd
import pickle
import logging
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import hashlib

class newDataStorageUtils:
    """
    Utility class for data pipeline storage operations.
    
    Handles serialization, file management, and logging for the pipeline.
    """
    
    def __init__(self, base_output_dir: Optional[str] = None, log_level: int = logging.INFO):
        """Initialize the dataStorageUtils class."""
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path.cwd()
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def _setup_logging(log_level: int) -> None:
        """Configure logging for the pipeline."""
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str, 
                    use_timestamp: bool = False, **kwargs) -> Path:
        """Save a DataFrame to a CSV file."""
        path = Path(filepath)
        
        if use_timestamp:
            base_name = path.stem
            extension = path.suffix
            timestamped_name = self.generate_timestamped_filename(base_name, extension)
            path = path.parent / timestamped_name
        
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, **kwargs)
        self.logger.info(f"CSV saved to: {path}")
        return path
    
    def load_from_csv(self, filepath: str, **kwargs) -> pd.DataFrame:
        """Load a DataFrame from a CSV file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        df = pd.read_csv(path, **kwargs)
        self.logger.info(f"CSV loaded from: {path} (shape: {df.shape})")
        return df
    
    def serialize_model(self, model: Any, path: str, metadata: Optional[Dict] = None) -> Path:
        """Serialize (save) a model object to disk using pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "wb") as file:
            pickle.dump(model, file)
        
        if metadata:
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            self.logger.info(f"Model metadata saved to: {metadata_path}")
        
        self.logger.info(f"Model serialized to: {path}")
        return path
    
    def deserialize_model(self, path: str) -> Any:
        """Deserialize (load) a model object from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, "rb") as file:
            model = pickle.load(file)
        
        self.logger.info(f"Model deserialized from: {path}")
        return model
    
    @staticmethod
    def generate_timestamped_filename(base_name: str, extension: str = ".csv") -> str:
        """Generate a timestamped filename."""
        if not isinstance(base_name, str) or not isinstance(extension, str):
            raise TypeError("Base name and extension must be strings")
        
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{base_name}_{timestamp}{extension}"
