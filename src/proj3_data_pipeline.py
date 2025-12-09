"""
Crime Research Data Pipeline - Class Definition For Data Pipelines

This module defines dataPipeline, an composition-based class that acts as a sort
of "master" class for the advanced OOP & refactoring implementation of the crime 
research data pipeline, relating to several other classes.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Advanced OOP with Inheritance & Polymorphism (Project 3)
"""

import os
import pandas as pd
import pickle
import logging
import json
import requests
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import hashlib
from src.proj3_data_ingestion_new import newDataIngestion 
from src.proj3_data_utilities_new import newDataStorageUtils
from src.proj3_data_cleaning_new import newDataCleaner
from src.proj3_data_transformation_new import newDataTransformation
from src.proj3_data_analysis_new import newDataAnalysis 

class DataPipeline:
    """
    Orchestrates the entire data processing pipeline using composition.
    
    This class demonstrates the "has-a" relationship (composition) by containing
    instances of other classes and coordinating their interactions.

    Composition relationships:
    - DataPipeline HAS-A dataIngestion (for loading data)
    - DataPipeline HAS-A dataCleaning (for cleaning data)
    - DataPipeline HAS-A dataTransformation (for transforming data)
    - DataPipeline HAS-A dataAnalysis (for analyzing data)
    - DataPipeline HAS-A dataStorageUtils (for saving results)
    """
    
    def __init__(self, 
                 ingestion: Optional[newDataIngestion] = None,
                 storage: Optional[newDataStorageUtils] = None,
                 verbose: bool = True):
        """
        Initialize the DataPipeline with optional component instances.
        
        This demonstrates DEPENDENCY INJECTION - components are injected rather
        than created internally, allowing for flexibility and testability.
        
        Args:
            ingestion (Optional[dataIngestion]): Data ingestion component
            storage (Optional[dataStorageUtils]): Storage utilities component
            verbose (bool): Whether to print pipeline progress
        """
        # Composition: Pipeline contains these objects
        self.ingestion = ingestion or newDataIngestion()
        self.storage = storage or newDataStorageUtils()
        self.verbose = verbose
        
        # Pipeline state
        self.data = None
        self.cleaner = None
        self.transformer = None
        self.analyzer = None
        self.pipeline_history = []
        
        self._log("DataPipeline initialized")
    
    def _log(self, message: str):
        """Log a pipeline operation."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.pipeline_history.append(log_entry)
        if self.verbose:
            print(f"[PIPELINE] {message}")
    
    def load_data(self, source, source_name: Optional[str] = None) -> 'DataPipeline':
        """
        Load data into the pipeline using a DataSource (polymorphic).
        
        Args:
            source: A DataSource object or filepath string
            source_name (Optional[str]): Name for the data source
            
        Returns:
            DataPipeline: Self for method chaining
        """
        if isinstance(source, str):
            # Assume it's a CSV filepath
            self.data = self.ingestion.load_csv(source)
            self._log(f"Loaded data from CSV: {source}")
        else:
            # Assume it's a DataSource object
            self.data = self.ingestion.load_from_source(source, source_name)
            self._log(f"Loaded data from {source.__class__.__name__}")
        
        return self
    
    def clean(self, **kwargs) -> 'DataPipeline':
        """
        Clean the loaded data.
        
        Args:
            **kwargs: Arguments to pass to cleaning operations
            
        Returns:
            DataPipeline: Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create cleaner (composition)
        self.cleaner = newDataCleaner(self.data, verbose=self.verbose)
        
        # Apply cleaning
        strategy = kwargs.get('strategy', 'mean')
        self.cleaner.handle_missing_values(strategy=strategy)
        
        # Update pipeline data
        self.data = self.cleaner.frame
        self._log(f"Data cleaned ({len(self.cleaner.processing_history)} operations)")
        
        return self
    
    def transform(self, scale_columns: Optional[List[str]] = None) -> 'DataPipeline':
        """
        Transform the data.
        
        Args:
            scale_columns (Optional[List[str]]): Columns to scale
            
        Returns:
            DataPipeline: Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create transformer (composition)
        self.transformer = newDataTransformation(self.data, verbose=self.verbose)
        
        # Apply transformations
        self.transformer.generate_features()
        if scale_columns:
            self.transformer.scale_features(scale_columns)
        
        # Update pipeline data
        self.data = self.transformer.frame
        self._log(f"Data transformed ({len(self.transformer.processing_history)} operations)")
        
        return self
    
    def analyze(self) -> 'DataPipeline':
        """
        Analyze the data.
        
        Returns:
            DataPipeline: Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create analyzer (composition)
        self.analyzer = newDataAnalysis(self.data, verbose=self.verbose)
        self.analyzer.process()
        
        self._log(f"Data analyzed ({len(self.analyzer.processing_history)} operations)")
        
        return self
    
    def save_results(self, filepath: str, use_timestamp: bool = True) -> 'DataPipeline':
        """
        Save the processed data.
        
        Args:
            filepath (str): Output file path
            use_timestamp (bool): Whether to add timestamp to filename
            
        Returns:
            DataPipeline: Self for method chaining
        """
        if self.data is None:
            raise ValueError("No data to save. Run pipeline first.")
        
        saved_path = self.storage.save_to_csv(self.data, filepath, use_timestamp=use_timestamp)
        self._log(f"Results saved to: {saved_path}")
        
        return self
    
    def run_full_pipeline(self, source, output_path: str, 
                          clean_strategy: str = 'mean',
                          scale_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Run the complete pipeline in one call.
        
        Args:
            source: Data source (filepath or DataSource object)
            output_path (str): Where to save results
            clean_strategy (str): Cleaning strategy to use
            scale_columns (Optional[List[str]]): Columns to scale
            
        Returns:
            pd.DataFrame: Processed data
        """
        self._log("Starting full pipeline execution")
        
        # Execute pipeline stages using composition
        self.load_data(source)
        self.clean(strategy=clean_strategy)
        self.transform(scale_columns=scale_columns)
        self.analyze()
        self.save_results(output_path)
        
        self._log("Full pipeline execution completed")
        
        return self.data
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the entire pipeline execution.
        
        Returns:
            Dict: Summary of all components and operations
        """
        summary = {
            'pipeline_steps': len(self.pipeline_history),
            'data_shape': self.data.shape if self.data is not None else None,
            'ingestion': str(self.ingestion),
            'cleaning_operations': len(self.cleaner.processing_history) if self.cleaner else 0,
            'transformation_operations': len(self.transformer.processing_history) if self.transformer else 0,
            'analysis_operations': len(self.analyzer.processing_history) if self.analyzer else 0
        }
        return summary
    
    def __str__(self) -> str:
        """Return a string representation of the pipeline."""
        lines = [
            "DataPipeline (Composition Pattern)",
            "=" * 70,
            "Components (HAS-A relationships):",
            f"  • Ingestion: {self.ingestion}",
            f"  • Storage: {self.storage}",
            f"  • Cleaner: {'Initialized' if self.cleaner else 'Not used'}",
            f"  • Transformer: {'Initialized' if self.transformer else 'Not used'}",
            f"  • Analyzer: {'Initialized' if self.analyzer else 'Not used'}",
            "=" * 70,
            f"Pipeline Steps Executed: {len(self.pipeline_history)}",
            f"Current Data Shape: {self.data.shape if self.data is not None else 'No data loaded'}",
        ]
        
        if self.pipeline_history:
            lines.append("\nRecent Pipeline History:")
            for entry in self.pipeline_history[-5:]:
                lines.append(f"  {entry}")
        
        return "\n".join(lines)
