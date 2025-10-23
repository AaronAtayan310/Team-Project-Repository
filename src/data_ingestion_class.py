import os
from typing import Dict, Any, Optional
import pandas as pd
import requests

class dataIngestion:
    ''' 
    This class loads data from various sources including 
    CSV files and REST APIs

    Attributes:
        default_timeout (int): Default timeout for API requests in seconds.
        data_sources (list): List of successfully loaded data sources.
    '''

    def __init__(self, default_timeout: int = 10, track_sources: bool = True):
        '''
        Initialize the dataIngestion class

        Args:
            default_timeout(int): Default timeout for API requests in seconds. Must be positive integer
                                  Defaults to 10 seconds.
            track_sources(bool): Whether to track uploaded sources or not. Default = True
        
        Raises:
            TypeError: If default_timeout is not an integer and if track_sources is not a boolean
            ValueError: If default_timeout is not a positive integer
        '''
        if not isinstance(default_timeout, int):
            raise TypeError("default_timeout must be an integer")
        if default_timeout <= 0:
            raise ValueError("default_timeout must be a positive integer")
        if not isinstance(track_sources, bool):
            raise TypeError("track_sources must be a boolean")
        
        self._default_timeout = default_timeout
        self._track_sources = track_sources
        self._data_sources =[]

    @property
    def default_timeout(self) -> int:
        ''' Gets default_timeout for API requests'''
        return self._default_timeout
    
    @default_timeout.setter
    def default_timeout(self, value: int):
        '''
        Set the default timeout for API requests

        Args:
            value (int): New timeout value in seconds.
        
        Raises:
            TypeError: If value is not an integer.
            ValueError: If value is not positive.
        '''
        if not isinstance(value, int):
            raise TypeError("default_timeout must be an integer")
        if value <= 0:
            raise ValueError("default_timeout must be a positive integer")
        
        self._default_timeout = value
    
    @property
    def track_sources(self) -> bool:
        '''Get whether source tracking is enabled'''
        return self._track_sources
    
    @track_sources.setter
    def track_sources(self, value: bool):
        '''
        Set whether to track data sources
        '''
        if not isinstance(value, bool):
            raise TypeError("track_sources must be a boolean")
        self._track_sources = value
    
    def load_csv(filepath: str) -> pd.DataFrame:
        '''
        Load a CSV file into a pandas DataFrame.

        Args:
            filepath (str): Path to the CSV file.

        Returns:
            pd.DataFrame: Loaded data as a DataFrame.

        Raises:
            TypeError: If file is not a string
            FileNotFoundError: If the file does not exist.
        '''
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