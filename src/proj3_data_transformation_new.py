"""
Crime Research Data Pipeline - Class Definition For Refactored Data Transformation

This module defines newDataTransformation, an abstract derived class which is also 
a refactored implementation of the dataTransformation class from the earlier crime 
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
from sklearn.preprocessing import StandardScaler
from .proj3_data_processor import dataProcessor 

class newDataTransformation(dataProcessor):
    '''
    Class for transforming DataFrames through scaling and feature engineering
    Inherits from DataProcessor and specializes in feature transformation operations
    '''

    def __init__(self, frame: pd.DataFrame, verbose: bool = False):
        ''' 
        Initialize the data transformer
        Args:
            frame (pd.DataFrame): The DataFrame to transform
            verbose (bool): If True, print transformation information
        '''
        super().__init__(frame, verbose)
        self.scalers = {}
        self._log_operastion("Transformer initialized")

    def process(self) -> 'newDataTransformation':
        '''
        Perform default transformation: generate features
        Returns:
            newDataTransformation: Self for method chaining
        '''
        self._log_operation("Starting default process")
        self.generate_features()
        
        return self
    
    def validate(self) -> bool:
        ''' 
        Validates that the DataFrame has no infinite values and appropriate dtypes
        Returns:
            bool: True if DataFrame is valid for transformation
        '''
        has_no_inf = not np.isinf(self._frame.select_dtypes(include=[np.number])).any().any()
        has_numeric = len(self.frame.select_dtypes(include=[np.number]).columns) > 0
        is_valid = has_no_inf and has_numeric

        if self._verbose:
            print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
            if not has_no_inf:
                print("  - Infinite values detected")
            if not has_numeric:
                print("  - No numeric columns for transformation")
        
        return is_valid
    
    def scale_features(self, columns: List[str]) -> 'newDataTransformation':
        ''' 
        Scale the features of the DataFrame using StandardScaler
        Args:
            columns(List[str]): Columns to scale
        Returns:
            newDataTransformation: Self for method chaining
        '''
        df = self.frame.copy()
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])

        #Storing scaler for future use:
        self._scalers[tuple(columns)] = scaler

        self._frame = df
        self._log_operations(f"Scaled features: {columns}")

        return self
    
    def generate_features(self) -> 'newDataTransformation':
        ''' 
        Generate new derived features based on existing data
        Returns:
            newDataTransformation: Self for method chaining
        '''
        df = self._frame.copy()
        features_added = []

        if "value" in df.columns and "count" in df.columns:
            df["value_per_count"] = df["value"] / (df["count"] + 1e-9)
            features_added.append("value_per_count")
        
        self._frame = df

        if features_added:
            self._log_operation(f"Generated features: {features_added}")
        else:
            self._log_operation("No features generated (required columns not found)")
        
        return self
    
    def __str__(self) -> str:
        """Return a string representation of the dataTransformation object."""
        lines = [
            "dataTransformation (inherits from DataProcessor)",
            "=" * 60,
            f"Current Shape: {self._frame.shape[0]} rows × {self._frame.shape[1]} columns",
            f"Scaled Feature Sets: {len(self._scalers)}",
            f"Operations Performed: {len(self.processing_history)}",
            "=" * 60
        ]
        
        if self.processing_history:
            lines.append("Processing History:")
            for operation in self.processing_history[-5:]:
                lines.append(f"  • {operation}")
        
        return "\n".join(lines)
