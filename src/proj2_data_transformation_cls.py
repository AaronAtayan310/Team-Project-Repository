"""
Crime Research Data Pipeline - Core Data Transformation Class

This module defines the DataTransformation class.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: OOP Class Implementation (Project 2)
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
from typing import List, Union 


class DataTransformation:
    """
    This class allows for transforming dataframes into scaled or more feature-complete versions.

    Attributes:
        frame (pd.DataFrame): A pandas dataframe containing the data we are working with, holding relevant data
    """
    def __init__(self, frame: pd.DataFrame):
        """
        Initialize an object of the DataTransformation class.

        Args:
            frame: A pandas dataframe containing the data we are working with, holding relevant data
        
        Raises:
            TypeError: If the input 'frame' is not a pandas DataFrame.
        """
        self._frame = None 
        self.frame = frame

    @property
    def frame(self) -> pd.DataFrame:
        """
        Gets the dataframe we are working with.
        """
        return self._frame

    @frame.setter
    def frame(self, val: pd.DataFrame):
        """
        Sets the value of the dataframe we are working with.

        Args:
            val (pd.DataFrame): The value we are trying to assign to the dataframe

        Raises:
            TypeError: If val is anything other than a pandas dataframe
        """
        if not isinstance(val, pd.DataFrame):
            raise TypeError('Data to transform must be in DataFrame format, no other format is acceptable.')
        self._frame = val

    def scale_features(self, columns: List[str]):
        """
        Scales the features of the dataframe we are working with using StandardScaler (Z-score scaling).

        Args:
            columns (list[str]): Columns to scale. Must contain numeric data.
        
        Raises:
            KeyError: If any column in the list is not found in the DataFrame.
        """
        missing_cols = [col for col in columns if col not in self.frame.columns] # ensure columns exist before copying
        if missing_cols:
            raise KeyError(f"Columns not found in DataFrame: {missing_cols}")

        non_numeric = [col for col in columns if not pd.api.types.is_numeric_dtype(self.frame[col])] # quick validation check
        if non_numeric:
            raise TypeError(f"Columns must contain numeric data: {non_numeric}")
            
        df = self.frame.copy()
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        self.frame = df

    def generate_features(self):
        """
        Generate new derived features based on the data we are working with, update the dataframe to include them, 
        specifically generating 'value_per_count' if 'value' and 'count' columns exist.
        """
        df = self.frame.copy()
        if "value" in df.columns and "count" in df.columns:
            df["value_per_count"] = df["value"].astype(float) / (df["count"].astype(float) + 1e-9)
        self.frame = df

    def __str__(self) -> str:
        """
        Returns a string representation of the DataTransformation object (the current state of the data being transformed).

        Returns:
            str: A readable description of the object.
        """
        summary = f"DataTransformation object summary (Shape: {self.frame.shape}):\n"
        return summary + str(self.frame)

    def __repr__(self) -> str:
        """
        Returns a developer-targeted string representation of the DataTransformation object.

        Returns:
            str: A development-useful description showing the class name and the dataframe shape
        """
        return f"DataTransformation(frame=pd.DataFrame(shape={self.frame.shape}))"
