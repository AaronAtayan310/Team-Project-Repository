"""
Crime Research Data Pipeline - Class Definition For Refactored Data Cleaning

This module defines newDataCleaner, an abstract derived class which is also a 
refactored implementation of the dataCleaner class from the earlier crime 
research data pipeline core classes implementation.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

#ABSTRACT DERIVED CLASS
import pandas as pd
import numpy as np
from typing import Optional, List, Union
from src.proj3_data_processor import dataProcessor 

class newDataCleaner(dataProcessor):
    '''
    Class for cleaning and preprocessing Pandas dataframes

    Provides methods for missing values, standardizing column names, 
    normalizing text, removing outliers, and other common tasks
    
    Attributes:
        df (pd.DataFrame): The DataFrame being cleaned
        original_shape (tuple): The shape of the original DataFrame.
        cleaning_history (list): A log of cleaning operations performed.
    '''

    def __init__(self, df: pd.DataFrame, verbose: bool = False):
        '''
        Initialize the data cleaner with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to clean.
            verbose (bool): If True, print information about cleaning operations
        Raises:
            TypeError: If df is not a pandas DataFrame
            ValueError: If df is empty 
        '''
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        super().__init__(df, verbose)
        self._original_shape = df.shape
        self._log_operation("newDataCleaner Initalized")
    
    @property
    def df(self) -> pd.DataFrame:
        '''
        Get the current DataFrame
        '''
        return self._df
    
    @df.setter
    def df(self, value: pd.DataFrame):
        '''
        Set the DataFrame with validation
        '''
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Value must be a pandas DataFrame")
        self._df = value

    @property
    def original_shape(self) -> tuple:
        '''
        Get the original shape of the DataFrame
        '''
        return self._original_shape
    
    @property
    def cleaning_history(self) -> List[str]:
        '''
        Get the history of cleaning operations
        '''
        return self.processing_history
        
    @property
    def verbose(self) -> bool:
        '''
        Get verbose setting
        '''
        return self._verbose
    
    @verbose.setter
    def verbose(self, value: bool):
        '''
        Set verbose setting
        '''
        if not isinstance(value, bool):
            raise TypeError("Verbose must be a boolean")
        self._verbose = value

    def _log_operation(self, operation: str):
        '''
        log a cleaning operation to the history
        '''
        self._cleaning_history.append(operation)
        if self._verbose:
            print(f"[newDataCleaner] {operation}")

    def handle_missing_values(self, strategy: str = "mean", 
                              columns: Optional[List[str]] = None) -> 'newDataCleaner':
        '''
        Handle missing values in the DataFrame using a given strategy.

        Args:
            strategy (str): Method to handle missing values
            columns (Optional[List[str]]): specific columns to apply strategy to. If None, applies to all columns
        Returns:
            newDataCleaner: Self for method chaining
        Raises:
            ValueError: If strategy is invalid
        '''
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
    
    def normalize_text_column(self, column: str, remove_special_chars: bool = False) -> 'newDataCleaner':
        '''
        Normalize the text in a specified column

        Args:
            column (str): Column to normalize
            remove_special_chars (bool): If True, remove special characters
        Returns:
            newDataCleaner: Self for method chaining
        Raises:
            ValueError: If column doesn't exist in DataFrame
        '''
        if column not in self._df.columns:
            raise ValueError(f"Column '{column}' not found")
        
        self._df[column] = self._df[column].astype(str).str.lower().str.strip()

        if remove_special_chars:
            self._df[column] = self._df[column].str.replace(r'[^a-z0-9\s]', '', regex = True)

        self._log_operation(f"Normalized text column '{column}'")
        return self
    
    def __str__(self) -> str:
        '''
        Returns a string representation of the newDataCleaner object.

        Returns:
            str: Formatted summary
        '''

        missing_values = self._df.isnull().sum().sum()
        missing_pct = (
            missing_values / (self._df.shape[0] * self._df.shape[1])) * 100 if self._df.size > 0 else 0
        
        lines = [
            "newDataCleaner Summary",
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
    
    def process(self) -> 'newDataCleaner':
        '''
        Perform default cleaning process: handle missing values with mean strat.
        
        Returns:
            newDataCleaner: Self for method chaining
        '''
        self._log_operation("Starting default cleaning process")
        self.handle_missing_values(strategy = 'mean')
        return self
    
    def validate(self) -> bool:
        '''
        Validate that the DataFrame is in a clean state

        Returns:
            bool: True if no missing values and no duplicate rows
        '''
        has_no_missing = self._frame.isnull().sum().sum() == 0
        has_no_duplicates = not self._frame.duplicated().any()
        is_valid = has_no_missing and has_no_duplicates

        if self._verbose:
            print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
            if not has_no_missing:
                print(f"  - Missing values detected: {self._frame.isnull().sum().sum()}")
            if not has_no_duplicates:
                print(f"  - Duplicate rows detected: {self._frame.duplicated().sum()}")
        
        return is_valid
