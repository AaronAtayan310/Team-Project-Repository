"""
Crime Research Data Pipeline - Core Data Analysis Class

This module defines the DataAnalysis class.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: OOP Class Implementation (Project 2)
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Iterator, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class DataAnalysis:
    """
    This class uses pandas dataframes holding data of interest and inegrates them with instance methods allowing analysis of the data.

    Attributes:
        frame (pd.DataFrame): A pandas dataframe containing the data we are working with.
        __described (pd.DataFrame): Description of the dataframe we are working with, a
        dataframe showing mean, deviation, so forth for the different columns
    """
    def __init__(self, frame):
        """
        Initialize the an object of the DataAnalysis class.

        Args:
            frame: A pandas dataframe containing the data we are working with, holding relevant data
        """
        self._frame = None
        self.frame = frame
        self.__described = self.frame.describe(include=np.number) # initial calculation of __described (will be overwritten by setter, but necessary for constructor)

    @property
    def frame(self):
        """
        Gets the dataframe we are working with.
        """
        return self._frame

    @frame.setter
    def frame(self, val: pd.DataFrame):
        """
        Sets the value of the dataframe we are working with.

        Args:
            val: The value we are trying to assign to the dataframe

        Raises:
            ValueError: If val is anything other than a pandas dataframe
        """
        if not(isinstance(val, pd.DataFrame)):
            raise ValueError('Data to analyze must be in DataFrame format, no other format is acceptable.')
        self._frame = val
        self.__described = self._frame.describe(include=np.number) # ensure __described is updated whenever the frame changes

    @property
    def described(self) -> pd.DataFrame:
        """
        Gets the description of the dataframe we are working with (a dataframe showing mean, deviation, so forth for the different columns).
        """
        return self.__described
        
    def run_regression(self, y: pd.Series) -> LinearRegression:
        """
        Fit a simple linear regression model.

        Args:
            y (pd.Series): Target variable

        Returns:
            model (LinearRegression): Trained regression model
        """
        X = self.frame.select_dtypes(include=np.number).dropna(axis=1) # select numeric columns from the instance frame to use as features (X)
        
        if X.empty or len(X) != len(y):
             raise ValueError("Feature DataFrame (self.frame) must contain compatible numeric columns and match the length of the target variable (y).")

        model = LinearRegression()
        model.fit(X, y) 
        return model
        
    def evaluate_model(self, model: LinearRegression, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a trained regression model using Mean Squared Error (MSE).

        Args:
            model (LinearRegression): Trained model
            y_test (pd.Series): Test targets

        Returns:
            dict: Dictionary with evaluation metrics
        """
        X_predict = self.frame.select_dtypes(include=np.number).dropna(axis=1) # select the same set of numeric features used for training

        if len(X_predict) != len(y_test):
             raise ValueError("Feature DataFrame (self.frame) used for prediction must match the length of the test target variable (y_test).")

        predictions = model.predict(X_predict)
        mse = mean_squared_error(y_test, predictions)
        return {"mse": mse}
        
    def calculate_missing_data(self) -> pd.Series:
        """
        Calculate the percentage of missing data in each column of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to analyze

        Returns:
            pd.Series: A Series containing the percentage of missing data per column

        Raises:
            TypeError: If 'df' is not a pandas DataFrame
        """
        return (self.frame.isnull().sum() / len(self.frame)) * 100
        
    def compute_crime_rate_by_year(self, population_col: str = "population") -> pd.DataFrame:
        """
        Compute annual crime rates per 100,000 people.

        Args:
            population_col (str): Column name representing population data

        Returns:
            yearly_data (pd.DataFrame): DataFrame with columns ['year', 'crime_count', 'crime_rate']
        """
        df = self.frame.copy()
        
        if "date" not in df.columns:
            raise ValueError("The dataset must contain a 'date' column.")
        if population_col not in df.columns:
            raise ValueError(f"Missing '{population_col}' column for population data.")

        df["date"] = pd.to_datetime(df["date"], errors='coerce') # convert date column to datetime objects
        df = df.dropna(subset=['date']) 

        df["year"] = df["date"].dt.year
        yearly_data = df.groupby("year").agg(
            crime_count=("crime_type", "count"),
            population=(population_col, "first") # 'first' for keeping population data integrity per year
        ).reset_index()

        yearly_data["crime_rate"] = (yearly_data["crime_count"] / yearly_data["population"]) * 100000
        return yearly_data
        
    def top_crime_types(self, n: int = 10) -> pd.DataFrame:
        """
        Identify the top N most frequent crime types.

        Args:
            n (int): The number of top crime types to return

        Returns:
            pd.DataFrame: DataFrame with columns ['crime_type', 'count']
        """
        if "crime_type" not in self.frame.columns:
            raise ValueError("The dataset must include a 'crime_type' column.")
        
        crime_counts = (
            self.frame["crime_type"] 
            .value_counts()
            .head(n)
            .reset_index(name='count') # name the count column explicitly
            .rename(columns={"index": "crime_type"}) 
        )
        return crime_counts
        
    def find_high_crime_areas(self, area_col: str = "neighborhood") -> pd.DataFrame:
        """
        Identify the areas with the highest number of reported crimes.

        Args:
            area_col (str): The name of the column representing geographic areas, defaulted to "neighborhood"

        Returns:
            area_stats (pd.DataFrame): DataFrame of areas sorted by descending crime count
        """
        if area_col not in self.frame.columns:
            raise ValueError(f"'{area_col}' column not found in dataset.")
        
        area_stats = (
            self.frame.groupby(area_col)
            .size()
            .reset_index(name="crime_count")
            .sort_values(by="crime_count", ascending=False)
        )
        return area_stats

    def __str__(self):
        """
        Returns a string representation of the DataAnalysis object (the source and description dataframes, in string form).

        Returns:
            str: A readable description of the object
        """
        source = str(self.frame)
        description = str(self.described)
        return f"Source of data:\n\n{source}\n\nDescription of data:\n\n{description}"

    def __repr__(self):
        """
        Returns a developer-targeted string representation of the DataAnalysis object, summarizing key information on the releated dataframe.

        Returns:
            str: A development-useful representation helpful for tasks like debugging
        """
        return f"DataAnalysis(frame_shape={self.frame.shape}, described_shape={self.described.shape})"
