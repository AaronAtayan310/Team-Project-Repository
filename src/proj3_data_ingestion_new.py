"""
Crime Research Data Pipeline - Class Definition For Refactored Data Ingestion

This module defines the NewDataIngestion class, a refactored implementation
of the DataIngestion class from the earlier crime research data pipeline core
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
from src.proj3_data_sources import CSVDataSource


class NewDataIngestion:
    """
    Refactored class for core data ingestion features in the data pipeline.
    """

    def __init__(self, default_timeout: int = 10, track_sources: bool = True):
        """
        Initialize an object of the NewDataIngestion class.
        
        Args:
            default_timeout (int): Default timeout for API requests
            track_sources (bool): Whether to track loaded sources
        """
        if not isinstance(default_timeout, int):
            raise TypeError("default_timeout must be an integer")
        if default_timeout <= 0:
            raise ValueError("default_timeout must be a positive integer")
        if not isinstance(track_sources, bool):
            raise TypeError("track_sources must be a boolean")
        
        self._default_timeout = default_timeout
        self._track_sources = track_sources
        self._data_sources = []
        self._loaded_data = {}  # Cache loaded DataFrames
    
    @property
    def default_timeout(self) -> int:
        """
        Get default timeout for API requests.
        """
        return self._default_timeout
    
    @default_timeout.setter
    def default_timeout(self, value: int):
        """
        Set the default timeout for API requests.

        Args:
            value (int): What we want the default timeout to match after setter usage.
        """
        if not isinstance(value, int) or value <= 0:
            raise ValueError("default_timeout must be a positive integer")
        self._default_timeout = value
    
    def load_from_source(self, source, source_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load data using a AbstractDataSource object (polymorphic method).
        
        This method demonstrates polymorphism: it accepts any AbstractDataSource subclass
        and calls its load() method, which behaves differently based on the actual type.
        
        Args:
            source (AbstractDataSource): A AbstractDataSource object (CSV, API, Database, etc.)
            source_name (Optional[str]): Optional name to cache the loaded data
            
        Returns:
            pd.DataFrame: Loaded data
            
        Raises:
            TypeError: If source is not a AbstractDataSource instance
        """
        # In a real implementation, would check isinstance(source, AbstractDataSource)
        if not hasattr(source, 'load') or not hasattr(source, 'validate_source'):
            raise TypeError("source must be a AbstractDataSource object with load() and validate_source() methods")
        
        # Validate before loading (polymorphic call)
        if not source.validate_source():
            raise ValueError(f"Invalid data source: {source}")
        
        # Load data (polymorphic call - behavior depends on actual source type)
        df = source.load()
        
        # Track the source
        if self._track_sources:
            self._data_sources.append({
                'source_type': source.__class__.__name__,
                'metadata': source.metadata,
                'loaded_at': datetime.now().isoformat(),
                'rows': len(df),
                'columns': len(df.columns)
            })
        
        # Cache the data if name provided
        if source_name:
            self._loaded_data[source_name] = df
        
        return df
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Convenience method to load CSV using CSVDataSource.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        # Create a CSVDataSource and delegate to it (composition in action)
        csv_source = CSVDataSource(filepath)
        return self.load_from_source(csv_source, source_name=filepath)
    
    def fetch_api_data(self, url: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch JSON data from a REST API endpoint.
        
        Args:
            url (str): The API URL
            params (dict, optional): Query parameters
            timeout (int, optional): Request timeout
            
        Returns:
            dict: Parsed JSON response
        """
        if not isinstance(url, str):
            raise TypeError("URL must be a string")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dictionary or None")
        
        actual_timeout = timeout if timeout is not None else self._default_timeout
        
        response = requests.get(url, params=params, timeout=actual_timeout)
        response.raise_for_status()
        data = response.json()
        
        if self._track_sources:
            self._data_sources.append({
                'type': 'api',
                'source': url,
                'params': params,
                'status_code': response.status_code
            })
        
        return data
    
    @staticmethod
    def validate_csv_path(file_path: str) -> bool:
        """
        Validate whether a given file path points to an existing CSV file.

        Args: 
            file_path (str): The actual file path used in validating a given csv file
        """
        if not isinstance(file_path, str):
            raise TypeError("File path must be a string")
        return os.path.isfile(file_path) and file_path.lower().endswith(".csv")
    
    def get_loaded_data(self, source_name: str) -> Optional[pd.DataFrame]:
        """
        Retrieve previously loaded data from cache.
        
        Args:
            source_name (str): Name of the cached data source
            
        Returns:
            Optional[pd.DataFrame]: Cached DataFrame or None
        """
        return self._loaded_data.get(source_name)
    
    def clear_sources(self):
        """
        Clear the list of tracked data sources and cached data.
        """
        self._data_sources.clear()
        self._loaded_data.clear()
    
    def __str__(self) -> str:
        """
        Return a string representation of the NewDataIngestion object.
        """
        sources_count = len(self._data_sources)
        cached_count = len(self._loaded_data)
        tracking_status = "enabled" if self._track_sources else "disabled"
        return (f"NewDataIngestion (timeout={self._default_timeout}s, "
                f"tracking={tracking_status}, sources_loaded={sources_count}, "
                f"cached={cached_count})")
