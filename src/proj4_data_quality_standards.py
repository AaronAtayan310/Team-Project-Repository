"""
Crime Research Data Pipeline - Data Quality Standards and Validation

This module defines the DataQualityStandards class which manages temporal
data quality rules, validation thresholds, and quality scoring for crime
datasets following FBI UCR and other reporting standards.

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, ClassVar
from datetime import datetime, timedelta
from enum import Enum


class QualityLevel(Enum):
    """Enumeration of quality levels for data assessment."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class ReportingStandard(Enum):
    """Enumeration of crime reporting standards."""
    FBI_UCR = "FBI_UCR"  # FBI Uniform Crime Reporting
    NIBRS = "NIBRS"  # National Incident-Based Reporting System
    LOCAL = "LOCAL"  # Local jurisdiction standards


class DataQualityStandards:
    """
    Manages temporal data quality rules and validation for crime datasets.
    
    Demonstrates:
    - New business logic layer for domain-specific rules
    """
    
    # Class-level quality thresholds
    QUALITY_THRESHOLDS: ClassVar[Dict[str, Dict[str, float]]] = {
        'completeness': {
            'excellent': 0.98,
            'good': 0.95,
            'acceptable': 0.90,
            'poor': 0.80,
            'critical': 0.70
        },
        'accuracy': {
            'excellent': 0.98,
            'good': 0.95,
            'acceptable': 0.90,
            'poor': 0.85,
            'critical': 0.80
        },
        'timeliness_days': {
            'excellent': 7,
            'good': 14,
            'acceptable': 30,
            'poor': 60,
            'critical': 90
        },
        'consistency': {
            'excellent': 0.98,
            'good': 0.95,
            'acceptable': 0.90,
            'poor': 0.85,
            'critical': 0.80
        }
    }
    
    # Required fields for different reporting standards
    REQUIRED_FIELDS: ClassVar[Dict[str, List[str]]] = {
        'FBI_UCR': [
            'incident_id', 'date', 'crime_type', 'location', 
            'jurisdiction', 'reported_date'
        ],
        'NIBRS': [
            'incident_id', 'date', 'crime_type', 'location',
            'jurisdiction', 'victim_count', 'offender_count',
            'reported_date', 'cleared_date'
        ],
        'LOCAL': [
            'incident_id', 'date', 'crime_type', 'location'
        ]
    }
    
    def __init__(self, jurisdiction: str, reporting_standard: str = 'FBI_UCR'):
        """
        Initialize DataQualityStandards for a specific jurisdiction.
        
        Args:
            jurisdiction: The jurisdiction name (e.g., "Maryland", "Baltimore")
            reporting_standard: The reporting standard to use (FBI_UCR, NIBRS, LOCAL)
        """
        self._jurisdiction = jurisdiction
        self._reporting_standard = ReportingStandard[reporting_standard]
        self._created_at = datetime.now()
        
        # Validation history
        self._validation_history: List[Dict[str, Any]] = []
    
    @property
    def jurisdiction(self) -> str:
        """Get the jurisdiction name."""
        return self._jurisdiction
    
    @property
    def reporting_standard(self) -> ReportingStandard:
        """Get the reporting standard."""
        return self._reporting_standard
    
    @property
    def validation_history(self) -> List[Dict[str, Any]]:
        """Get the history of validation operations."""
        return self._validation_history.copy()
    
    @classmethod
    def get_threshold(cls, metric: str, level: str) -> float:
        """
        Get a specific quality threshold value.
        
        Args:
            metric: The metric name (completeness, accuracy, etc.)
            level: The quality level (excellent, good, acceptable, poor, critical)
            
        Returns:
            float: The threshold value
            
        Raises:
            ValueError: If metric or level is invalid
        """
        if metric not in cls.QUALITY_THRESHOLDS:
            raise ValueError(f"Unknown metric: {metric}. Available: {list(cls.QUALITY_THRESHOLDS.keys())}")
        
        if level not in cls.QUALITY_THRESHOLDS[metric]:
            raise ValueError(f"Unknown level: {level}. Available: {list(cls.QUALITY_THRESHOLDS[metric].keys())}")
        
        return cls.QUALITY_THRESHOLDS[metric][level]
    
    @classmethod
    def get_required_fields(cls, standard: str) -> List[str]:
        """
        Get required fields for a reporting standard.
        
        Args:
            standard: The reporting standard name
            
        Returns:
            List of required field names
        """
        return cls.REQUIRED_FIELDS.get(standard, cls.REQUIRED_FIELDS['LOCAL'])
    
    def validate_data_freshness(self, data: pd.DataFrame, date_col: str = 'date') -> Dict[str, Any]:
        """
        Validate the freshness of crime data (temporal validation).
        
        Args:
            data: DataFrame to validate
            date_col: Name of the date column
            
        Returns:
            Dict: Freshness validation results
        """
        if date_col not in data.columns:
            return {
                'valid': False,
                'quality_level': QualityLevel.CRITICAL.value,
                'error': f"Date column '{date_col}' not found",
                'days_old': None
            }
        
        # Convert to datetime
        dates = pd.to_datetime(data[date_col], errors='coerce')
        most_recent = dates.max()
        
        if pd.isna(most_recent):
            return {
                'valid': False,
                'quality_level': QualityLevel.CRITICAL.value,
                'error': "No valid dates found",
                'days_old': None
            }
        
        # Calculate age
        days_old = (datetime.now() - most_recent).days
        
        # Determine quality level based on thresholds
        if days_old <= self.QUALITY_THRESHOLDS['timeliness_days']['excellent']:
            quality = QualityLevel.EXCELLENT
        elif days_old <= self.QUALITY_THRESHOLDS['timeliness_days']['good']:
            quality = QualityLevel.GOOD
        elif days_old <= self.QUALITY_THRESHOLDS['timeliness_days']['acceptable']:
            quality = QualityLevel.ACCEPTABLE
        elif days_old <= self.QUALITY_THRESHOLDS['timeliness_days']['poor']:
            quality = QualityLevel.POOR
        else:
            quality = QualityLevel.CRITICAL
        
        result = {
            'valid': quality != QualityLevel.CRITICAL,
            'quality_level': quality.value,
            'days_old': days_old,
            'most_recent_date': most_recent.strftime('%Y-%m-%d'),
            'threshold_excellent': self.QUALITY_THRESHOLDS['timeliness_days']['excellent'],
            'threshold_acceptable': self.QUALITY_THRESHOLDS['timeliness_days']['acceptable'],
        }
        
        self._validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'freshness',
            'result': result
        })
        
        return result
    
    def validate_completeness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data completeness (missing values assessment).
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dict: Completeness validation results
        """
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        completeness_score = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0.0
        
        # Determine quality level
        if completeness_score >= self.QUALITY_THRESHOLDS['completeness']['excellent']:
            quality = QualityLevel.EXCELLENT
        elif completeness_score >= self.QUALITY_THRESHOLDS['completeness']['good']:
            quality = QualityLevel.GOOD
        elif completeness_score >= self.QUALITY_THRESHOLDS['completeness']['acceptable']:
            quality = QualityLevel.ACCEPTABLE
        elif completeness_score >= self.QUALITY_THRESHOLDS['completeness']['poor']:
            quality = QualityLevel.POOR
        else:
            quality = QualityLevel.CRITICAL
        
        result = {
            'valid': quality != QualityLevel.CRITICAL,
            'quality_level': quality.value,
            'completeness_score': completeness_score,
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': (missing_cells / total_cells * 100) if total_cells > 0 else 0.0,
        }
        
        self._validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'completeness',
            'result': result
        })
        
        return result
    
    def validate_required_fields(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that data contains required fields for the reporting standard.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            Dict: Field validation results
        """
        required = self.REQUIRED_FIELDS[self._reporting_standard.value]
        present_fields = set(data.columns)
        required_fields = set(required)
        
        missing_fields = required_fields - present_fields
        extra_fields = present_fields - required_fields
        
        valid = len(missing_fields) == 0
        
        result = {
            'valid': valid,
            'required_fields': list(required_fields),
            'present_fields': list(present_fields),
            'missing_fields': list(missing_fields),
            'extra_fields': list(extra_fields),
            'compliance_score': len(required_fields & present_fields) / len(required_fields) if required_fields else 1.0,
        }
        
        self._validation_history.append({
            'timestamp': datetime.now().isoformat(),
            'validation_type': 'required_fields',
            'result': result
        })
        
        return result
    
    def calculate_quality_score(self, data: pd.DataFrame, date_col: str = 'date') -> Dict[str, Any]:
        """
        Calculate comprehensive data quality score.
        
        Args:
            data: DataFrame to assess
            date_col: Name of the date column
            
        Returns:
            Dict: Comprehensive quality assessment
        """
        # Run all validations
        freshness = self.validate_data_freshness(data, date_col)
        completeness = self.validate_completeness(data)
        fields = self.validate_required_fields(data)
        
        # Calculate composite score
        scores = []
        
        if freshness['valid']:
            # Convert quality level to numeric score
            quality_map = {
                'excellent': 1.0,
                'good': 0.85,
                'acceptable': 0.70,
                'poor': 0.55,
                'critical': 0.40
            }
            scores.append(quality_map.get(freshness['quality_level'], 0.5))
        
        if completeness['valid']:
            scores.append(completeness['completeness_score'])
        
        if fields['valid']:
            scores.append(fields['compliance_score'])
        
        overall_score = sum(scores) / len(scores) if scores else 0.0
        
        # Determine overall quality level
        if overall_score >= 0.95:
            overall_quality = QualityLevel.EXCELLENT
        elif overall_score >= 0.85:
            overall_quality = QualityLevel.GOOD
        elif overall_score >= 0.75:
            overall_quality = QualityLevel.ACCEPTABLE
        elif overall_score >= 0.65:
            overall_quality = QualityLevel.POOR
        else:
            overall_quality = QualityLevel.CRITICAL
        
        return {
            'overall_score': overall_score,
            'overall_quality': overall_quality.value,
            'freshness': freshness,
            'completeness': completeness,
            'field_compliance': fields,
            'timestamp': datetime.now().isoformat(),
            'jurisdiction': self._jurisdiction,
            'reporting_standard': self._reporting_standard.value,
        }
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all validation operations.
        
        Returns:
            Dict: Validation summary statistics
        """
        if not self._validation_history:
            return {
                'total_validations': 0,
                'by_type': {},
                'latest_validation': None
            }
        
        by_type = {}
        for validation in self._validation_history:
            vtype = validation['validation_type']
            by_type[vtype] = by_type.get(vtype, 0) + 1
        
        return {
            'total_validations': len(self._validation_history),
            'by_type': by_type,
            'latest_validation': self._validation_history[-1],
            'jurisdiction': self._jurisdiction,
            'reporting_standard': self._reporting_standard.value,
        }
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return (
            f"DataQualityStandards(jurisdiction='{self._jurisdiction}', "
            f"standard={self._reporting_standard.value})"
        )
    
    def __str__(self) -> str:
        """User-friendly representation."""
        return (
            f"Data Quality Standards for {self._jurisdiction} "
            f"({self._reporting_standard.value} compliance)"
        )
