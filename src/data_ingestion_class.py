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

    