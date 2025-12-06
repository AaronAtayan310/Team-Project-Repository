#ABSTRACT Derived Class
from pathlib import Path
import pandas as pd
from datetime import datetime
from abc import abc, abstractmethod
from src.data_source import dataSource

class CSVDataSource(dataSource):
    '''
    Concrete implementation for loading data from CSV files
    Demonstrates polymorphic behavior
    '''

    def __init__(self, filepath: str):
        '''
        Initialize CSV data source
        Args:
            filepath(str): Path to the CSV file
        '''
        super().__init__()
        self.filepath = filepath
        self._source_metadata['type'] = 'csv'
        self._source_metadata['filepath'] = filepath

    def validate_source(self) -> bool:
        '''
        Validate that the CSV file exists and is readable

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
