"""
Crime Research Data Pipeline - Class Definition For API Data

This module defines APIDataSource, an abstract derived class.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

#ABSTRACT DERIVED CLASS
from pathlib import Path
import pandas as pd
from datetime import datetime
from abc import abc, abstractmethod
from .proj3_data_source import dataSource
from typing import Any, Dict, List, Optional, Union

class APIDataSource(dataSource):
    '''
    Concrete implementation for loading data from REST APIs
    '''

    def __init__(self, url: str, params: Optional[Dict] = None):
        '''
        Initialize API data source

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
        Validate that the URL is formed correctly

        Returns:
            bool: True if valid
        '''
        is_valid = self.url.startswith(('http://', 'https://'))
        self._source_metadata['validated'] = is_valid
        return is_valid
    
    def load(self) -> pd.DataFrame:
        '''
        Load data from API endpoint

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
