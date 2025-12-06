#ABSTRACT Derived Class
from pathlib import Path
import pandas as pd
from datetime import datetime
from abc import abc, abstractmethod
from src.data_source import dataSource

class databaseDataSource(dataSource):
    '''
    Concrete implementation for loading data from databases
    '''
    def __init__(self, connection_string: str, query: str):
        '''
        Initialize database data source
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
        Validate that connection string and query are provided

        Returns:
            bool: True if valid
        '''
        is_valid = bool(self.connection_string and self.query)
        self._source_metadata['validated'] = is_valid
        
        return is_valid
    
    #WIP
