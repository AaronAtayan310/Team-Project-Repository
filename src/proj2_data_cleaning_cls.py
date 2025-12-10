"""
Crime Research Data Pipeline - Core Data Cleaning Class

This module defines the DataCleaner class.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: OOP Class Implementation (Project 2)
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Union

class DataCleaner:
    """
    Class for cleaning and preprocessing Pandas dataframes, providing
    methods for missing values, standardizing column names, 
    normalizing text, removing outliers, and other common tasks.
    
    Attributes:
        df (pd.DataFrame): The DataFrame being cleaned
        original_shape (tuple): The shape of the original DataFrame
        cleaning_history (list): A log of cleaning operations performed
    """

    def __init__(self, df: pd.DataFrame, verbose: bool = False):
        """
        Initialize the DataCleaner object with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to clean
            verbose (bool): If True, print information about cleaning operations
            
        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If df is empty 
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        self._df = df.copy()
        self._original_shape = df.shape
        self._cleaning_history = []
        self._verbose = verbose
    
    @property
    def df(self) -> pd.DataFrame:
        """
        Get the current DataFrame.
        """
        return self._df
    
    @df.setter
    def df(self, value: pd.DataFrame):
        """
        Set the DataFrame with validation.
        """
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Value must be a pandas DataFrame")
        self._df = value

    @property
    def original_shape(self) -> tuple:
        """
        Get the original shape of the DataFrame.
        """
        return self._original_shape
    
    @property
    def cleaning_history(self) -> List[str]:
        """
        Get the history of cleaning operations.
        """
        return self._cleaning_history.copy()
    
    @property
    def verbose(self) -> bool:
        """
        Get verbose setting.
        """
        return self._verbose
    
    @verbose.setter
    def verbose(self, value: bool):
        """
        Set verbose setting.
        """
        if not isinstance(value, bool):
            raise TypeError("Verbose must be a boolean")
        self._verbose = value

    def _log_operation(self, operation: str):
        """
        Log a cleaning operation to the history.
        """
        self._cleaning_history.append(operation)
        if self._verbose:
            print(f"[DataCleaner] {operation}")

    def handle_missing_values(self, strategy: str = "mean", columns: Optional[List[str]] = None) -> 'DataCleaner':
        """
        Handle missing values in the DataFrame using a given strategy.

        Args:
            strategy (str): Method to handle missing values
            columns (Optional[List[str]]): specific columns to apply strategy to. If None, applies to all columns
            
        Returns:
            DataCleaner: Self for method chaining
            
        Raises:
            ValueError: If strategy is invalid
        """
        valid_strategies = ['mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill']
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid Strategy. Choose from {valid_strategies}")
        
        target_df = self._df[columns] if columns else self._df
        missing_before = self._df.isnull().sum().sum()

        if strategy == "mean":
            if columns:
                self._df[columns] = self._df[columns].fillna(self._df[columns].mean(numeric_only=True))
            else:
                self._df.fillna(self._df.mean(numeric_only=True), inplace=True)
        
        elif strategy == "median":
            if columns:
                self._df[columns] = self._df[columns].fillna(self._df[columns].median(numeric_only=True))
            else:
                self._df.fillna(self._df.median(numeric_only=True), inplace=True)
        
        elif strategy == "mode":
            if columns:
                for col in columns:
                    if not self._df[col].mode().empty:
                        self._df[col].fillna(self._df[col].mode()[0], inplace=True)
            else:
                for col in self._df.columns:
                    if not self._df[col].mode().empty:
                        self._df[col].fillna(self._df[col].mode()[0], inplace=True)
        
        elif strategy == "drop":
            self._df.dropna(subset=columns, inplace=True)
        
        elif strategy == "forward_fill":
            if columns:
                self._df[columns] = self._df[columns].fillna(method='ffill')
            else:
                self._df.fillna(method='ffill', inplace=True)
        
        elif strategy == "backward_fill":
            if columns:
                self._df[columns] = self._df[columns].fillna(method='bfill')
            else:
                self._df.fillna(method='bfill', inplace=True)

        missing_after = self._df.isnull().sum().sum()
        cols_msg = f" in columns {columns}" if columns else ""
        self._log_operation(f"Handled missing values using '{strategy}' strategy{cols_msg}")
        return self
    
    def normalize_text_column(self, column: str, remove_special_chars: bool = False) -> 'DataCleaner':
        """
        Normalize the text in a specified column.

        Args:
            column (str): Column to normalize
            remove_special_chars (bool): If True, remove special characters
            
        Returns:
            DataCleaner: Self for method chaining
            
        Raises:
            ValueError: If column doesn't exist in DataFrame
        """
        if column not in self._df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        self._df[column] = self._df[column].astype(str).str.lower().str.strip()

        if remove_special_chars:
            self._df[column] = self._df[column].str.replace(r'[^a-z0-9\s]', '', regex = True)

        self._log_operation(f"Normalized text column '{column}'")
        return self
    
    def __str__(self) -> str:
        """
        Returns a string representation of the DataCleaner object.

        Returns:
            str: Formatted summary
        """

        missing_values = self._df.isnull().sum().sum()
        missing_pct = (
            missing_values / (self._df.shape[0] * self._df.shape[1])) * 100 if self._df.size > 0 else 0
        
        lines = [
            "DataCleaner Summary",
            "=" * 50,
            f"Current Shape: {self._df.shape[0]} rows × {self._df.shape[1]} columns",
            f"Original Shape: {self._original_shape[0]} rows × {self._original_shape[1]} columns",
            f"Missing Values: {missing_values} ({missing_pct:.2f}%)",
            f"Operations Performed: {len(self._cleaning_history)}",
            "=" * 50
        ]

        if self._cleaning_history:
            lines.append("Cleaning History:")
            for i, operation in enumerate(self._cleaning_history, 1):
                lines.append(f"  {i}. {operation}")
        else:
            lines.append("No cleaning operations performed")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """
        Developer-targeted string representation of the DataCleaner object.

        Returns:
            msg (str): A concise string including key info on the state of the DataCleaner object
        """
        msg = f"DataCleaner(df_shape={self._df.shape}, " + f"original_shape={self._original_shape}, " + f"operations={len(self._cleaning_history)}, " + f"verbose={self._verbose})"
        return msg
