"""
Crime Research Data Pipeline - Class Definition For Refactored Data Utilities

This module defines the NewDataStorageUtils class, a refactored implementation
of the DataStorageUtils class from the earlier crime research data pipeline core
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

class NewDataStorageUtils:
    """
    Refactored utility class for data pipeline storage operations, handling serialization, 
    file management, and logging for the pipeline.
    """
    
    def __init__(self, base_output_dir: Optional[str] = None, log_level: int = logging.INFO):
        """
        Initialize an object of the NewDataStorageUtils class.
        """
        self.base_output_dir = Path(base_output_dir) if base_output_dir else Path.cwd()
        self.base_output_dir.mkdir(parents=True, exist_ok=True)
        
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def _setup_logging(log_level: int) -> None:
        """
        Configure logging for the pipeline.

        Args:
            log_level (int): A number representing the level that logging should occur at
        """
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    def save_to_csv(self, df: pd.DataFrame, filepath: str, use_timestamp: bool = False, **kwargs) -> Path:
        """
        Save a DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The DataFrame containing the information we want saved to a CSV
            filepath (str): The filepath needed to locate the csv file that will hold the info
            use_timestamp (bool): Whether or not to generate a usage timestamp, defaulted to False
            **kawrgs: Addtional arguments as needed

        Returns:
            path (Path): A Path object representing how to find the csv
        """
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
        """
        Load a DataFrame from a CSV file.

        Args:
            filepath (str): The file path used to load the CSV data into a dataframe
            **kawrgs: Addtional arguments as needed

        Returns:
            df (pd.DataFrame): The dataframe holding the information retrieved from the csv
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        df = pd.read_csv(path, **kwargs)
        self.logger.info(f"CSV loaded from: {path} (shape: {df.shape})")
        return df
    
    def serialize_model(self, model: Any, path: str, metadata: Optional[Dict] = None) -> Path:
        """
        Serialize (save) a model object to disk using pickle.

        Args:
            model (Any): The model to work with
            path (str): The filepath used to identify the model
            metadata (Optional[Dict]): Metadata chararcterizing the model (optional, defaults to None)

        Returns:
            path (Path): A path object represeting how to find the model after saving
        """
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
        """
        Deserialize (load) a model object from disk.

        Args:
            path (str): The needed file path to load the model object

        Returns:
            model (Any): The model itself loaded from the disk using the given file path
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, "rb") as file:
            model = pickle.load(file)
        
        self.logger.info(f"Model deserialized from: {path}")
        return model
    
    @staticmethod
    def generate_timestamped_filename(base_name: str, extension: str = ".csv") -> str:
        """
        Generate a timestamped filename.

        Args:
            base_name (str): The basic name (no extension yet) the timestamped file should be represented by
            extension (str): The file extension specific to this timestamped file, defaulted to ".csv"

        Returns:
            str: The timestamped filename with the basic name and appropiate file path
        """
        if not isinstance(base_name, str) or not isinstance(extension, str):
            raise TypeError("Base name and extension must be strings")
        
        if not extension.startswith('.'):
            extension = f'.{extension}'
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return f"{base_name}_{timestamp}{extension}"
