"""
Crime Research Data Pipeline - Abstract Base Class Definitions

This module defines the abstract interfaces that all concrete data processing
and data source subclasses must implement.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
import logging


class AbstractDataProcessor(ABC):
    """
    Abstract base class for all data processing classes, defining 
    the common interface that all data processors must implement
    which ensures consistent behavior across classes.

    Attributes:
        frame (pd.DataFrame): The DataFrame being processed
        processing_history (List[str]): Log of performed operations
        verbose (bool): Whether to print processing information or not
    """

    def __init__(self, frame: pd.DataFrame, verbose: bool = False):
        """
        Initalize the AbstractDataProcessor with a DataFrame.

        Args:
            frame (pd.DataFrame): The DataFrame to process
            verbose (bool): Prints processing info if True
        
        Raises:
            TypeError: if frame isn't a pandas DataFrame
            ValueError: if frame is empty
        """
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if frame.empty:
            raise ValueError("DataFrame cannot be empty")
        
        self._frame = frame.copy()
        self._processing_history = []
        self._verbose = verbose
    
    @property
    def frame(self) -> pd.DataFrame:
        """
        Get the current DataFrame.
        """
        return self._frame
    
    @frame.setter
    def frame (self, value: pd.DataFrame):
        """
        Sets DataFrame with validation.
        """
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Value must be a pandas DataFrame")
        self._frame = value
    
    @property
    def processing_history(self) -> List[str]:
        """
        Gets the history of processing operations.
        """
        return self._processing_history.copy()
    
    def _log_operations(self, operation: str):
        """
        Log operation to history.
        
        Args:
            operation (str): Description of operation
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {operation}"
        self._processing_history.append(log_entry)
        if self._verbose:
            print(f"[{self.__class__.__name__}] {operation}")

    #Abstract methods
    @abstractmethod
    def process(self) -> 'AbstractDataProcessor':
        """
        Abstract method to perform the main processing operation,
        must be implemented by all subclasses.

        Returns:
            AbstractDataProcessor: Self for method chaining
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Abstract method to validate the DataFrame state,
        must be implemented by all subclasses.

        Returns:
            bool: True if validation passes
        """
        pass

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current DataFrame state.

        Returns:
            Dict: Summary statistics
        """
        return{
            'shape': self._frame.shape,
            'columns': list(self._frame.columns),
            'dtypes': self.frame.dtypes.to_dict(),
            'missing_values': self._frame.isnull().sum().to_dict(),
            'operations_count': len(self._processing_history)
        }


class AbstractDataSource(ABC):
    """
    Abstract base class for all data source types, defining the common interface for 
    loading data from various sources, enabling polymorphic behavior where different 
    sources can be used interchangeably.
    
    Attributes:
        _source_metadata (Dict): Metadata about the loaded data source
    """

    def __init__(self):
        """
        Initialize the AbstractDataSource.
        """
        self._source_metadata = {}
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the data source.
        """
        return self._source_metadata.copy()
    
    #Abstract method(s)
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Abstract method to load data from the source,
        must be implemented by all subclasses.

        Returns:
            pd.DataFrame: Loaded data
        """
        pass

    @abstractmethod
    def validate_source(self) -> bool:
        """
        Abstract method to validate the data source before loading,
        must be implemented by all subclasses.

        Returns:
            bool: True if source is valid
        """
        pass

    def get_source_info(self) -> str:
        """
        Get a string representation of the source info.

        Returns:
            str: Source info
        """
        source_type = self.__class__.__name__
        metadata_str = ", ".join(f"{k}={v}" for k, v in self._source_metadata.items())
        return f"{source_type}({metadata_str})"
