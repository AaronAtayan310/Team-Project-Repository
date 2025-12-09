"""
Crime Research Data Pipeline - Class Definition For Refactored Data Analysis

This module defines newDataAnalysis, an abstract derived class.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

#ABSTRACT DERIVED CLASS
import pandas as pd
import numpy as np
from typing import Optional, List, Union
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from src.proj3_data_processor import dataProcessor

class newDataAnalysis(dataProcessor):
    ''' 
    Class for statistical analysis and machine learning on DataFrames.
    
    Inherits from DataProcessor and specializes in analytical operations.
    '''

    def __init__(self, frame: pd.DataFrame, verbose: bool = False):
        ''' 
        Initalize the data analyzer
        Args:
            frame (pd.DataFrame): The DataFrame to analyze
            verbose (bool): If True, print analysis information
        '''
        super().__init__(frame, verbose)
        self.__described = self._frame.describe()
        self._models = {} #This will store trained models
        self._log_operation("Data Analyzer initialized")

    @property
    def described(self) -> pd.DataFrame:
        ''' 
        Get description of DataFrame
        '''
        return self.__described
    
    def process(self) -> 'newDataAnalysis':
        ''' 
        Perform default analysis: update description stats
        Returns:
            newDataAnalysis: Self for method chaining
        '''
        self._log_operation("Starting default analysis process")
        self.__described = self._frame.describe()

        return self
    
    def validate(self) -> bool:
        ''' 
        Validate that the DataFrame has sufficient data for analysis
        Returns:
            bool: True if DataFrame has enough rows and numeric columns
        '''
        has_enough_rows = len(self._frame) >= 2
        has_numeric = len(self._frame.select_dtypes(include=[np.number]).columns) > 0
        is_valid = has_enough_rows and has_numeric
        
        if self._verbose:
            print(f"Validation: {'PASSED' if is_valid else 'FAILED'}")
            if not has_enough_rows:
                print(f"  - Insufficient rows: {len(self._frame)} (need at least 2)")
            if not has_numeric:
                print("  - No numeric columns for analysis")
        
        return is_valid
    
    def run_regression(self, y: pd.Series, model_name: str = "default") -> LinearRegression:
        ''' 
        Fit a simple linear regression model
        Args:
            y (pd.Series): Target variable
            model_name (str): Name to store the model under
        
        Returns:
            LinearRegression: Trained regression model
        '''
        model = LinearRegression()
        model.fit(self._frame. y)
        self._models[model_name] = model
        self._log_operation(f"Trained regression model '{model_name}'")
        
        return model
    
    def evaluate_model(self, model: LinearRegression, y_test: pd.Series) -> Dict[str, float]:
        ''' 
        Evaluate a trained regression model using Mean Squared Error
        Args:
            model (LinearRegression): Trained model
            y_test (pd.Series): Test targets
        Returns:
            dict: Dictionary with evaluation metrics
        '''
        predictions = model.predict(self._frame)
        mse = mean_squared_error(y_test, predictions)
        self._log_operation(f"Model evaluated: MSE = {mse:.4f}")

        return {"mse": mse}

    def calculate_missing_data(self) -> pd.Series:
        ''' 
        Calculate the percentage of missing data in each column

        Returns:
            pd.Series: A series containing the percentage of missing data
        '''
        return (self._frame.isnull().sum() / len(self._frame)) * 100
    
    def compute_crime_rate_by_year(self, population_col: str = "population") -> pd.DataFrame:
        """
        Compute annual crime rates per 100,000 people.
        
        Args:
            population_col (str): Column name representing population data
            
        Returns:
            pd.DataFrame: DataFrame with columns ['year', 'crime_count', 'crime_rate']
        """
        df = self._frame.copy()
        if "date" not in df.columns:
            raise ValueError("The dataset must contain a 'date' column.")
        if population_col not in df.columns:
            raise ValueError(f"Missing '{population_col}' column for population data.")
        
        df["year"] = pd.to_datetime(df["date"]).dt.year
        yearly_data = df.groupby("year").agg(
            crime_count=("crime_type", "count"),
            population=(population_col, "mean")
        ).reset_index()
        yearly_data["crime_rate"] = (yearly_data["crime_count"] / yearly_data["population"]) * 100000
        
        self._log_operation("Computed crime rates by year")
        return yearly_data
    
    def top_crime_types(self, n: int = 10) -> pd.DataFrame:
        ''' 
        Identify the top N most frequeny crime types

        Args:
            n (int): The number of top crime types to return
        Returns:
            pd.DataFrame: DataFrame with column ['crime_type', 'count']
        '''
        if "crime_type" not in self._frame.columns:
            raise ValueError("The dataset must include a 'crime_type' column")
        
        result = (
            self._frame["crime_type"]
            .value_counts()
            .head(n)
            .reset_index()
            .rename(columns={"crime_type": "count", "index": "crime_type"})
        )

        self._log_operation(f"Identified top {n} crime types")

        return result
    
     def find_high_crime_areas(self, area_col: str = "neighborhood") -> pd.DataFrame:
        """
        Identify the areas with the highest number of reported crimes.
        
        Args:
            area_col (str): The name of the column representing geographic areas
            
        Returns:
            pd.DataFrame: DataFrame of areas sorted by descending crime count
        """
        if area_col not in self._frame.columns:
            raise ValueError(f"'{area_col}' column not found in dataset.")
        
        area_stats = (
            self._frame.groupby(area_col)
            .size()
            .reset_index(name="crime_count")
            .sort_values(by="crime_count", ascending=False)
        )
        
        self._log_operation(f"Analyzed high crime areas by '{area_col}'")
        return area_stats
    
    def __str__(self) -> str:
        """Return a string representation of the dataAnalysis object."""
        lines = [
            "dataAnalysis (inherits from DataProcessor)",
            "=" * 60,
            f"DataFrame Shape: {self._frame.shape[0]} rows × {self._frame.shape[1]} columns",
            f"Trained Models: {len(self._models)}",
            f"Operations Performed: {len(self.processing_history)}",
            "=" * 60,
            "\nDescriptive Statistics:",
            str(self.__described)
        ]
        
        return "\n".join(lines)