"""
Crime Research Data Pipeline - Core Data Ingestion Class

This module defines the DataIngestion class.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: OOP Class Implementation (Project 2)
"""

import os
from typing import Dict, Any, Optional, List
import pandas as pd
import requests


class DataIngestion:
    """ 
    This class loads data from various sources including CSV files and REST APIs.

    Attributes:
        default_timeout (int): Default timeout for API requests in seconds
        track_sources (bool): Whether data sources are being tracked
        data_sources (list): List of successfully loaded data sources
    """

    def __init__(self, default_timeout: int = 10, track_sources: bool = True):
        """
        Initialize an object of the DataIngestion class.

        Args:
            default_timeout(int): Default timeout for API requests in seconds. Must be a 
                                  positive integer, and is defaulted to 10 seconds
            track_sources(bool): Whether or not to track uploaded sources, defaulted to True
        
        Raises:
            TypeError: If default_timeout is not an integer and if track_sources is not a boolean
            ValueError: If default_timeout is not a positive integer
        """
        if not isinstance(default_timeout, int):
            raise TypeError("default_timeout must be an integer")
        if default_timeout <= 0:
            raise ValueError("default_timeout must be a positive integer")
        if not isinstance(track_sources, bool):
            raise TypeError("track_sources must be a boolean")
        
        self._default_timeout = default_timeout
        self._track_sources = track_sources
        self._data_sources: List[Dict[str, Any]] = []

    @property
    def default_timeout(self) -> int:
        """ 
        Gets default_timeout for API requests.
        """
        return self._default_timeout
    
    @default_timeout.setter
    def default_timeout(self, value: int):
        """
        Set the default timeout for API requests.

        Args:
            value (int): New timeout value in seconds
        
        Raises:
            TypeError: If value is not an integer
            ValueError: If value is not positive
        """
        if not isinstance(value, int):
            raise TypeError("default_timeout must be an integer")
        if value <= 0:
            raise ValueError("default_timeout must be a positive integer")
        
        self._default_timeout = value
    
    @property
    def track_sources(self) -> bool:
        """
        Get whether source tracking is enabled.
        """
        return self._track_sources
    
    @track_sources.setter
    def track_sources(self, value: bool):
        """
        Set whether to track data sources.
        """
        if not isinstance(value, bool):
            raise TypeError("track_sources must be a boolean")
        self._track_sources = value

    @property
    def data_sources(self) -> List[Dict[str, Any]]:
        """
        Gets the list of tracked data sources (to match the class Attribute docstring).
        """
        return self._data_sources
    
    def load_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load a CSV file into a pandas DataFrame.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded data as a DataFrame

        Raises:
            TypeError: If file is not a string
            FileNotFoundError: If the file does not exist
        """
        if not isinstance(filepath, str):
            raise TypeError("The filepath must be a string")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        df = pd.read_csv(filepath)

        if self._track_sources:
            self._data_sources.append({
                'type': 'csv',
                'source': filepath,
                'rows': len(df),
                'columns': len(df.columns)
            })

        return df
    
    def fetch_api_data(self, url: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch JSON data from a REST API endpoint.

        Args:
            url (str): the API URL
            params (Optional[Dict[str, Any]]): Query parameters to include in the request (optional, defaulted to None)
            timeout (Optional[int]): Request timeout in seconds (optional, defaulted to None)
        Returns:
            dict: Parsed JSON response
        Raises:
            TypeError: If URL is not a string or params is not a dict
            requests.RequestException: if the API request fails
        """
        if not isinstance(url, str):
            raise TypeError("URL must be a string")
        if params is not None and not isinstance(params, dict):
            raise TypeError("params must be a dictionary or None")
        
        actual_timeout = timeout if timeout is not None else self._default_timeout

        if timeout is not None and (not isinstance(timeout, int) or timeout <= 0): # type checking for if an override is provided
             raise ValueError("If provided, timeout must be a positive integer.")

        response = requests.get(url, params = params, timeout = actual_timeout)
        response.raise_for_status()
        data = response.json()

        if self._track_sources:
            if isinstance(data, list): # check if data is a list (common for JSON APIs) and get item count, else 1
                 item_count = len(data)
            elif isinstance(data, dict):
                 item_count = 1
            else:
                 item_count = 0

            self._data_sources.append({
                'type': 'api',
                'source': url,
                'params': params,
                'status_code': response.status_code,
                'items_retrieved': item_count
            })
        
        return data
    
    @staticmethod
    def validate_csv_path(file_path: str) -> bool:
        """
        Validate whether a given file path points to an existing CSV file.

        Args:
            file_path (str): The path to the file being validated
        Returns:
            bool: True if the file exists and has a '.csv' extension, False otherwise
        Raises:
            TypeError: If 'file_path' is not a string
        """
        if not isinstance(file_path, str):
            raise TypeError("File path must be a string")
        
        return os.path.isfile(file_path) and file_path.lower().endswith(".csv")
    

    def clear_sources(self):
        """
        Clears the list of tracked data sources.
        """
        self._data_sources.clear()

    def __str__(self) -> str:
        """
        Return a string representation of the DataIngestion object.

        Returns:
            str: A readable description of the object
        """
        sources_count = len(self._data_sources)
        tracking_status = "enabled" if self._track_sources else "disabled"
        return (f"DataIngestion (timeout= {self._default_timeout}s, "
                f"tracking= {tracking_status}, sources_loaded= {sources_count})")
    
    def __repr__(self) -> str:
        """
        Return a detailed string representation of the DataIngestion object.

        Returns:
            str: A string that could be used to recreate the DataIngestion object
        """
        return (f"DataIngestion(default_timeout={self._default_timeout}, "
                f"track_sources={self._track_sources})")
