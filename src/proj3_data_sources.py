"""
Crime Research Data Pipeline - Concrete Implementations Of Data Sources

This module contains specialized data source classes that inherit from 
AbstractDataSource and provide source-specific behavior.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

from .proj3_base_classes import AbstractDataSource
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import pandas as pd
from datetime import datetime
from abc import abc, abstractmethod


class APIDataSource(AbstractDataSource):
    '''
    Concrete implementation for loading data from REST APIs.
    '''

    def __init__(self, url: str, params: Optional[Dict] = None):
        '''
        Initialize an API data source.

        Args:
            url (str): API endpoint URL
            params (Optional[Dict]): Query parameters
        '''
        super().__init__()
        self.url = url
        self.params = params or {}
        self._source_metadata['type'] = 'api'
        self._source_metadata['url'] = url
    
    def validate_source(self) -> bool:
        '''
        Validate that the URL is formed correctly.

        Returns:
            bool: True if valid
        '''
        is_valid = self.url.startswith(('http://', 'https://'))
        self._source_metadata['validated'] = is_valid
        return is_valid
    
    def load(self) -> pd.DataFrame:
        '''
        Load data from API endpoint.

        Returns:
            pd.DataFrame: Loaded data

        Raises:
            ValueError: If URL is invalid
            ImportError: If requests library is not available
        '''
        if not self.validate_source():
            raise ValueError(f"Invalid API URL")

        try:
            import requests
        except ImportError:
            raise ImportError("'requests' library is required for API data sources")
        
        response = requests.get(self.url, params = self.params, timeout = 10)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            raise ValueError("API response must be JSON list or dict")
        
        self._source_metadata['rows_loaded'] = len(df)
        self._source_metadata['status_code'] = response.status_code
        self._source_metadata['load_time'] = datetime.now().isoformat()

        return df


class CSVDataSource(AbstractDataSource):
    '''
    Concrete implementation for loading data from CSV files,
    demonstrating polymorphic behavior.
    '''

    def __init__(self, filepath: str):
        '''
        Initialize a CSV data source.
        
        Args:
            filepath(str): Path to the CSV file
        '''
        super().__init__()
        self.filepath = filepath
        self._source_metadata['type'] = 'csv'
        self._source_metadata['filepath'] = filepath

    def validate_source(self) -> bool:
        '''
        Validate that the CSV file exists and is readable.

        Returns:
            bool: True if valid
        '''
        path = Path(self.filepath)
        is_valid = path.exists() and path.suffix.lower() == '.csv'
        self._source_metadata['validated'] = is_valid
        
        return is_valid

    def load(self) -> pd.DataFrame:
        '''
        Load data from CSV file.

        Returns:
            pd.DataFrame: Loaded data
        
        Raises:
            FileNotFoundError: If file doesn't exist
        '''
        if not self.validate_source():
            raise FileNotFoundError(f"CSV file not found: {self.filepath}")
        
        df = pd.read_csv(self.filepath)
        self._source_metadata['rows_loaded'] = len(df)
        self._source_metadata['columns_loaded'] = len(df.columns)
        self._source_metadata['load_time'] = datetime.now().isoformat()

        return df


class DatabaseDataSource(AbstractDataSource):
    '''
    Concrete implementation for loading data from databases.
    '''
    
    def __init__(self, connection_string: str, query: str):
        '''
        Initialize a database data source.
        
        Args:
            connection_string (str): Database connection string
            query (str): SQL query to execute
        '''
        super().__init__()
        self.connection_string = connection_string
        self.query = query
        self._source_metadata['type'] = 'database'

    def validate_source(self) -> bool:
        '''
        Validate that connection string and query are provided.

        Returns:
            bool: True if valid
        '''
        is_valid = bool(self.connection_string and self.query)
        self._source_metadata['validated'] = is_valid
        
        return is_valid

    def load(self) -> pd.DataFrame:
        '''
        Load data from a database.

        Returns:
            pd.DataFrame: Loaded data
        
        Raises:
            FILL-IN-BLANK
        '''
        if not self.validate_source():
            # raise FILL-IN-BLANK

        # INSERT IMPLEMENTATION HERE INCLUDING DEFINITION FOR df

        return df
