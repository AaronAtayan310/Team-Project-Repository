#ABSTRACT CLASS
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
import logging

class dataSource(ABC):
    '''
    Abstract base class for all data source types

    This class defines the common interface for loading data from various sources,
    enabling polymorphic behavior where different sources can be used interchangeably.
    
    Attributes:
        _source_metadata (Dict): Metadata about the loaded data source
    '''

    def __init__(self):
        '''
        Initialize the DataSource
        '''
        self._source_metadata = {}
    
    @property
    def metadata(self) -> Dict[str, Any]:
        '''
        Get metadata about the data source
        '''
        return self._source_metadata.copy()
    
    #Abstract method(s)
    @abstractmethod
    def load(self) -> pd.DataFrame:
        '''
        Abstract method to load data from the source
        must be implemented by all subclasses

        Returns:
            pd.DataFrame: Loaded data
        '''
        pass

    @abstractmethod
    def validate_source(self) -> bool:
        '''
        Abstract method to validate the data source before loading
        must be implemented by all subclasses

        Returns:
            bool: True if source is valid
        '''
        pass

    def get_source_info(self) -> str:
        '''
        Get a string representation of the source info

        Returns:
            str: Source info
        '''
        source_type = self.__class__.__name__
        metadata_str = ", ".join(f"{k}={v}" for k, v in self._source_metadata.items())
        return f"{source_type}({metadata_str})"
