"""
Crime Research Data Pipeline - Project 4 Demonstration

Comprehensive demonstration of all Project 4 enhanced features including:
- Enhanced base classes with UUID, temporal tracking, registries
- DataQualityStandards business logic
- Factory methods and composition patterns
- Advanced analytics and reporting
- Complete pipeline orchestration

Author: INST326 Crime Research Data Pipeline Project Team (Group 0203-SAV-ASMV)
Course: Object-Oriented Programming for Information Science
Institution: University of Maryland, College Park
Project: Capstone Integration & Testing (Project 4)
"""

import pandas as pd
import numpy as np
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add src directory to path so we can import the relevant code files
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))    

from src.proj4_data_source import DataSource
from src.proj4_data_processor import DataProcessor
from src.proj4_specialized_sources import CSVCrimeDataSource, APICrimeDataSource, DatabaseCrimeDataSource
from src.proj4_specialized_processors import CrimeDataAnalysis, CrimeDataCleaner, CrimeDataTransformation
from src.proj4_data_quality_standards import DataQualityStandards, QualityLevel, ReportingStandard
from src.proj4_data_ingestion import CrimeDataIngestion
from src.proj4_data_utilities import CrimeDataStorageUtils
from src.proj4_pipeline_manager import PipelineManager


def main():
    """
    Main demonstration following a single narrative flow.
    
    Scenario: You are a crime analyst for Maryland law enforcement.
    You need to process, clean, analyze, and report on recent crime data.
    """
    
    print("\n" + "="*80)
    print("CRIME RESEARCH DATA PIPELINE - PROJECT 4 COMPREHENSIVE DEMONSTRATION")
    print("="*80)
    print("\nScenario: Processing Maryland Crime Data for Monthly Report")
    print("="*80)
    
    print("\n[SETUP] Creating sample crime dataset...") # create sample data
    print("-" * 80)
    
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
    crime_types = ['Theft', 'Assault', 'Burglary', 'Robbery', 'Vandalism', 
                   'Drug Offense', 'Vehicle Theft', 'Fraud']
    locations = ['Downtown', 'Eastside', 'Westside', 'Northside', 'Southside', 
                'Suburbs', 'Industrial', 'Waterfront']
    neighborhood = ['District 1', 'District 2', 'District 3', 'District 4', 
                 'District 5', 'Bel Air', 'Edgewood', 'Abingdon']
    
    df = pd.DataFrame({
        'incident_id': range(1, 101),
        'date': np.random.choice(dates, 100),
        'crime_type': np.random.choice(crime_types, 100),
        'location': np.random.choice(locations, 100),
        'neighborhood': np.random.choice(neighborhood, 100),
        'jurisdiction': ['Maryland'] * 100,
        'reported_date': np.random.choice(dates, 100),
        'value': np.random.randint(100, 10000, 100),
        'count': np.random.randint(1, 5, 100),
        'population': [850000] * 100
    })
    
    # Introduce some data quality issues (realistic scenario)
    df.loc[np.random.choice(df.index, 5, replace=False), 'crime_type'] = None
    df.loc[np.random.choice(df.index, 3, replace=False), 'value'] = None
    
    # Save to temporary CSV
    temp_csv = Path("maryland_crime_data.csv")
    df.to_csv(temp_csv, index=False)
    
    print(f"✓ Created dataset with {len(df)} incidents")
    print(f"✓ Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"✓ Saved to: {temp_csv}")
    print(f"✓ Data quality issues: {df.isnull().sum().sum()} missing values")
    
    # ========== STEP 1: Data Quality Standards ==========
    print("\n\n" + "="*80)
    print("STEP 1: ESTABLISHING DATA QUALITY STANDARDS")
    print("="*80)
    print("\nBefore processing data, we need to establish quality benchmarks.")
    print("Maryland uses FBI Uniform Crime Reporting (UCR) standards.")
    print("-" * 80)
    
    standards = DataQualityStandards("Maryland", "FBI_UCR")
    
    print(f"\n✓ Quality Standards Initialized")
    print(f"  Jurisdiction: {standards.jurisdiction}")
    print(f"  Standard: {standards.reporting_standard.value}")
    
    print("\nQuality Thresholds:")
    print("  Completeness:")
    for level in ['excellent', 'good', 'acceptable']:
        threshold = DataQualityStandards.get_threshold('completeness', level)
        print(f"    - {level.capitalize()}: {threshold:.0%}")
    
    print("\n  Timeliness:")
    for level in ['excellent', 'good', 'acceptable']:
        days = DataQualityStandards.get_threshold('timeliness_days', level)
        print(f"    - {level.capitalize()}: {days} days")
    
    required_fields = DataQualityStandards.get_required_fields('FBI_UCR')
    print(f"\n  Required Fields ({len(required_fields)}): {', '.join(required_fields)}")
    
    # ========== STEP 2: Enhanced Data Sources ==========
    print("\n\n" + "="*80)
    print("STEP 2: CREATING ENHANCED DATA SOURCES")
    print("="*80)
    print("\nProject 4 sources have UUID tracking, temporal metadata, and registries.")
    print("-" * 80)
    
    # Create CSV source
    print("\nCreating CSV data source...")
    csv_source = CSVCrimeDataSource(str(temp_csv), name="Maryland_Monthly_Data")
    
    print(f"✓ Source Created: {csv_source.name}")
    print(f"  Type: {csv_source.__class__.__name__}")
    print(f"  UUID: {csv_source.source_id}")
    print(f"  Created: {csv_source.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Demonstrate registry
    print("\nRegistry Pattern - All sources are automatically tracked:")
    all_sources = DataSource.get_all_sources()
    print(f"✓ Total registered sources: {len(all_sources)}")
    for source in all_sources:
        print(f"  - {source.name} ({source.__class__.__name__})")
    
    # Load data and track
    print("\nLoading data with temporal tracking...")
    data = csv_source.load()
    
    print(f"✓ Data Loaded: {len(data)} rows × {len(data.columns)} columns")
    print(f"  Load count: {csv_source.load_count}")
    print(f"  Last loaded: {csv_source.last_loaded_at.strftime('%H:%M:%S')}")
    
    # Show load statistics
    stats = csv_source.get_load_statistics()
    print(f"\nLoad Statistics:")
    print(f"  Total loads: {stats['total_loads']}")
    print(f"  Success rate: {stats['success_rate']:.0%}")
    print(f"  Average duration: {stats['average_duration']:.4f}s")
    
    # Data lineage
    lineage = csv_source.get_data_lineage()
    print(f"\nData Lineage:")
    print(f"  Source ID: {lineage['source_id'][:16]}...")
    print(f"  Data checksum: {lineage['last_data_checksum'][:16]}...")
    print(f"  Data shape: {lineage['last_data_shape']}")
    
    # ========== STEP 3: Initial Quality Assessment ==========
    print("\n\n" + "="*80)
    print("STEP 3: INITIAL DATA QUALITY ASSESSMENT")
    print("="*80)
    print("\nAssessing data quality before processing...")
    print("-" * 80)
    
    initial_quality = standards.calculate_quality_score(data)
    
    print(f"\n✓ Overall Quality Score: {initial_quality['overall_score']:.2%}")
    print(f"✓ Overall Quality Level: {initial_quality['overall_quality']}")
    
    print("\nDetailed Assessment:")
    print(f"  Freshness: {initial_quality['freshness']['quality_level']}")
    print(f"    - Most recent: {initial_quality['freshness']['most_recent_date']}")
    print(f"    - Age: {initial_quality['freshness']['days_old']} days")
    
    print(f"\n  Completeness: {initial_quality['completeness']['quality_level']}")
    print(f"    - Score: {initial_quality['completeness']['completeness_score']:.2%}")
    print(f"    - Missing: {initial_quality['completeness']['missing_cells']} cells")
    
    print(f"\n  Field Compliance: {initial_quality['field_compliance']['compliance_score']:.2%}")
    print(f"    - Missing fields: {initial_quality['field_compliance']['missing_fields']}")
    
    # ========== STEP 4: Pipeline Manager Setup ==========
    print("\n\n" + "="*80)
    print("STEP 4: INITIALIZING PIPELINE MANAGER")
    print("="*80)
    print("\nThe PipelineManager orchestrates all components (composition pattern).")
    print("-" * 80)
    
    # Create output directory
    output_dir = Path("pipeline_output")
    output_dir.mkdir(exist_ok=True)
    
    # Initialize ingestion and storage components
    ingestion = CrimeDataIngestion(
        default_timeout=10,
        track_sources=True,
        quality_standards=standards
    )
    
    storage = CrimeDataStorageUtils(
        base_output_dir=str(output_dir),
        enable_versioning=True
    )
    
    # Create pipeline with components
    pipeline = PipelineManager(
        ingestion=ingestion,
        storage=storage,
        quality_standards=standards,
        verbose=True
    )
    
    print(f"\n✓ Pipeline Initialized")
    print(f"  Pipeline ID: {pipeline.pipeline_id}")
    print(f"  Created: {pipeline.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n  Composition (HAS-A relationships):")
    print(f"  ✓ Ingestion: {pipeline.ingestion.__class__.__name__}")
    print(f"    - ID: {pipeline.ingestion.ingestion_id[:16]}...")
    print(f"  ✓ Storage: {pipeline.storage.__class__.__name__}")
    print(f"    - ID: {pipeline.storage.storage_id[:16]}...")
    print(f"  ✓ Quality Standards: {pipeline.quality_standards.__class__.__name__}")
    
    # ========== STEP 5: Load Data into Pipeline ==========
    print("\n\n" + "="*80)
    print("STEP 5: LOADING DATA INTO PIPELINE")
    print("="*80)
    print("\nUsing factory method to create and load from source...")
    print("-" * 80)
    
    # Factory method to create source
    print("\nFactory Method: create_csv_source()")
    source = pipeline.create_csv_source(str(temp_csv), name="Maryland_Data")
    print(f"✓ Created source: {source.name}")
    
    # Load data
    print("\nLoading data with quality validation...")
    pipeline.load_data(source, validate_quality=True)
    
    print(f"✓ Data loaded into pipeline")
    print(f"  Shape: {pipeline.data.shape}")
    print(f"  Columns: {list(pipeline.data.columns)}")
    
    # Show ingestion statistics
    ingestion_stats = pipeline.ingestion.get_ingestion_statistics()
    print(f"\nIngestion Statistics:")
    print(f"  Total loads: {ingestion_stats['total_loads']}")
    print(f"  Success rate: {ingestion_stats['success_rate']:.0%}")
    
    # ========== STEP 6: Data Cleaning ==========
    print("\n\n" + "="*80)
    print("STEP 6: DATA CLEANING WITH QUALITY TRACKING")
    print("="*80)
    print("\nCreating cleaner and handling missing values...")
    print("-" * 80)
    
    # Factory method to create cleaner
    print("\nFactory Method: create_cleaner()")
    cleaner = pipeline.create_cleaner()
    print(f"✓ Created cleaner: {cleaner.name}")
    print(f"  Processor ID: {cleaner.processor_id[:16]}...")
    
    print(f"\nBefore cleaning:")
    print(f"  Missing values: {pipeline.data.isnull().sum().sum()}")
    
    # Clean using pipeline
    print("\nCleaning data (strategy: mean imputation)...")
    pipeline.clean(strategy='mean')
    
    print(f"\nAfter cleaning:")
    print(f"  Missing values: {pipeline.data.isnull().sum().sum()}")
    
    # Get cleaning impact
    impact = cleaner.get_cleaning_impact()
    print(f"\nCleaning Impact:")
    print(f"  Original missing: {impact['original_missing']}")
    print(f"  Missing resolved: {impact['missing_resolved']}")
    
    if 'quality_improvement' in impact:
        print(f"  Quality improvement: {impact['quality_improvement']:+.2%}")
    
    # Show performance
    perf = cleaner.get_performance_metrics()
    print(f"\nCleaner Performance:")
    print(f"  Operations: {perf['total_operations']}")
    print(f"  Total time: {perf['total_time_seconds']:.4f}s")
    print(f"  Average per operation: {perf['average_operation_time']:.4f}s")
    
    # ========== STEP 7: Data Transformation ==========
    print("\n\n" + "="*80)
    print("STEP 7: DATA TRANSFORMATION WITH FEATURE LINEAGE")
    print("="*80)
    print("\nGenerating new features and scaling...")
    print("-" * 80)
    
    # Factory method to create transformer
    print("\nFactory Method: create_transformer()")
    transformer = pipeline.create_transformer()
    print(f"✓ Created transformer: {transformer.name}")
    
    print(f"\nBefore transformation:")
    print(f"  Columns: {len(pipeline.data.columns)}")
    
    # Transform using pipeline
    print("\nTransforming data...")
    print("  - Generating features")
    print("  - Scaling numeric columns")
    pipeline.transform(scale_columns=['value', 'count'], generate_features=True)
    
    print(f"\nAfter transformation:")
    print(f"  Columns: {len(pipeline.data.columns)}")
    
    # Show feature lineage
    lineage = transformer.get_feature_lineage()
    print(f"\nFeature Lineage ({len(lineage)} new features):")
    for feature in lineage:
        print(f"  ✓ {feature['feature_name']}")
        print(f"    Formula: {feature['formula']}")
        print(f"    Derived from: {feature['derived_from']}")
        print(f"    Created: {feature['created_at']}")
    
    # Show scaler info
    scalers = transformer.scalers
    print(f"\nScaled Features: {len(scalers)} sets")
    for cols, info in scalers.items():
        print(f"  ✓ Columns: {list(cols)}")
        print(f"    Fitted: {info['fitted_at']}")
    
    # ========== STEP 8: Data Analysis ==========
    print("\n\n" + "="*80)
    print("STEP 8: CRIME DATA ANALYSIS")
    print("="*80)
    print("\nPerforming crime-specific analytics...")
    print("-" * 80)
    
    # Factory method to create analyzer
    print("\nFactory Method: create_analyzer()")
    analyzer = pipeline.create_analyzer()
    print(f"✓ Created analyzer: {analyzer.name}")
    
    # Analyze using pipeline
    print("\nPerforming analysis...")
    pipeline.analyze()
    
    # Crime-specific analytics
    print("\n--- Crime Rate by Year ---")
    crime_rates = analyzer.compute_crime_rate_by_year()
    print(crime_rates.to_string(index=False))
    
    print("\n--- Top 5 Crime Types ---")
    top_crimes = analyzer.top_crime_types(n=5)
    print(top_crimes.to_string(index=False))
    
    print("\n--- High Crime Areas (Top 5) ---")
    high_crime = analyzer.find_high_crime_areas()
    print(high_crime.head().to_string(index=False))
    
    # Analysis with quality
    print("\n--- Analysis Quality Assessment ---")
    analysis_quality = analyzer.get_analysis_with_quality()
    print(f"Models trained: {analysis_quality['model_count']}")
    print(f"Data completeness: {analysis_quality['data_quality']['completeness_score']:.2%}")
    
    if 'quality_assessment' in analysis_quality:
        print(f"Overall quality: {analysis_quality['quality_assessment']['overall_quality']}")
    
    # ========== STEP 9: Save Results ==========
    print("\n\n" + "="*80)
    print("STEP 9: SAVING PROCESSED RESULTS")
    print("="*80)
    print("\nSaving with timestamps and versioning...")
    print("-" * 80)
    
    # Save main results
    print("\nSaving processed crime data...")
    output_file = output_dir / "processed_crime_data.csv"
    pipeline.save_results(str(output_file), use_timestamp=True)
    
    # Check what was saved
    saved_files = storage.list_saved_files()
    print(f"\n✓ Files saved: {len(saved_files)}")
    for file_info in saved_files:
        print(f"  - {Path(file_info['filepath']).name}")
        print(f"    Size: {file_info['file_size_bytes']:,} bytes")
        print(f"    Checksum: {file_info['checksum'][:16]}...")
    
    # Storage statistics
    storage_stats = storage.get_storage_statistics()
    print(f"\nStorage Statistics:")
    print(f"  Total saves: {storage_stats['total_saves']}")
    print(f"  Success rate: {storage_stats['success_rate']:.0%}")
    print(f"  Total data written: {storage_stats['total_mb_written']:.2f} MB")
    
    # ========== STEP 10: Pipeline Analytics ==========
    print("\n\n" + "="*80)
    print("STEP 10: ADVANCED PIPELINE ANALYTICS")
    print("="*80)
    print("\nGenerating comprehensive pipeline reports...")
    print("-" * 80)
    
    # Processing schedule
    print("\n--- Processing Schedule ---")
    schedule = pipeline.get_processing_schedule()
    for stage in schedule:
        print(f"✓ {stage['stage'].upper()}: {stage['status']}")
        print(f"  Component: {stage['component']}")
        if 'operations' in stage:
            print(f"  Operations: {stage['operations']}")
    
    # Data quality report
    print("\n--- Final Data Quality Report ---")
    quality_report = pipeline.get_data_quality_report()
    print(f"Pipeline ID: {quality_report.get('pipeline_id', 'N/A')[:16]}...")
    print(f"Data shape: {quality_report['current_data_shape']}")
    
    if 'quality_assessment' in quality_report:
        final_quality = quality_report['quality_assessment']
        print(f"\nFinal Quality Score: {final_quality['overall_score']:.2%}")
        print(f"Quality Level: {final_quality['overall_quality']}")
        
        # Compare to initial
        improvement = final_quality['overall_score'] - initial_quality['overall_score']
        print(f"\nQuality Improvement: {improvement:+.2%}")
    
    # Pipeline lineage
    print("\n--- Pipeline Lineage ---")
    lineage = pipeline.get_pipeline_lineage()
    print(f"Pipeline Steps: {lineage['pipeline_steps']}")
    print(f"Processors Used: {len(lineage['processors'])}")
    for proc in lineage['processors']:
        print(f"  ✓ {proc['processor_type']}: {proc['operations']} operations")
    
    # Performance metrics
    print("\n--- Performance Metrics ---")
    metrics = pipeline.get_performance_metrics()
    print(f"Total Pipeline Steps: {metrics['total_pipeline_steps']}")
    
    if 'processors' in metrics and metrics['processors']:
        print("\nProcessor Performance:")
        for proc_name, proc_metrics in metrics['processors'].items():
            if 'total_time_seconds' in proc_metrics:
                print(f"  {proc_name}:")
                print(f"    Operations: {proc_metrics['total_operations']}")
                print(f"    Time: {proc_metrics['total_time_seconds']:.4f}s")
                print(f"    Avg: {proc_metrics['average_operation_time']:.4f}s/op")
    
    # ========== STEP 11: Pipeline Validation ==========
    print("\n\n" + "="*80)
    print("STEP 11: PIPELINE VALIDATION")
    print("="*80)
    print("\nValidating pipeline integrity...")
    print("-" * 80)
    
    validation = pipeline.validate_pipeline()
    
    print(f"\n✓ Pipeline Valid: {validation['valid']}")
    print(f"  Issues: {len(validation['issues'])}")
    print(f"  Warnings: {len(validation['warnings'])}")
    
    if validation['issues']:
        print("\n  Issues Found:")
        for issue in validation['issues']:
            print(f"    ❌ {issue}")
    
    if validation['warnings']:
        print("\n  Warnings:")
        for warning in validation['warnings']:
            print(f"    ⚠ {warning}")
    
    if validation['valid'] and not validation['warnings']:
        print("\n  ✓ Pipeline is healthy and ready for production!")
    
    # ========== STEP 12: Export for Audit ==========
    print("\n\n" + "="*80)
    print("STEP 12: EXPORT PIPELINE STATE FOR AUDIT")
    print("="*80)
    print("\nExporting complete pipeline state and configuration...")
    print("-" * 80)
    
    # Export state
    state = pipeline.export_pipeline_state()
    state_file = output_dir / "pipeline_state.json"
    
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    
    print(f"\n✓ Pipeline State Exported: {state_file}")
    print(f"  Pipeline ID: {state['pipeline_id']}")
    print(f"  Steps executed: {state['pipeline_steps']}")
    print(f"  Active processors: {len(state['processors'])}")
    
    # Export config
    config = pipeline.export_pipeline_config()
    config_file = output_dir / "pipeline_config.json"
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"\n✓ Pipeline Config Exported: {config_file}")
    print(f"  Pipeline type: {config['pipeline_type']}")
    print(f"  Version: {config['version']}")
    
    # ========== STEP 13: Demonstrate Polymorphism ==========
    print("\n\n" + "="*80)
    print("STEP 13: POLYMORPHISM DEMONSTRATION")
    print("="*80)
    print("\nAll sources and processors share common interfaces...")
    print("-" * 80)
    
    print("\n--- Polymorphic Sources ---")
    print("Creating different source types with same interface:")
    
    sources = [
        CSVCrimeDataSource(str(temp_csv), name="CSV_Source"),
        APICrimeDataSource("https://api.example.com/crimes", name="API_Source"),
    ]
    
    for source in sources:
        print(f"\n{source.__class__.__name__}:")
        print(f"  - validate_source(): {source.validate_source()}")
        print(f"  - source_id: {source.source_id[:16]}...")
        print(f"  - metadata keys: {list(source.metadata.keys())}")
    
    print("\n--- Polymorphic Processors ---")
    print("All processors implement process() and validate():")
    
    # Get all registered processors
    all_processors = DataProcessor.get_all_processors()
    for proc in all_processors:
        print(f"\n{proc.__class__.__name__}:")
        print(f"  - processor_id: {proc.processor_id[:16]}...")
        print(f"  - validate(): {proc.validate()}")
        print(f"  - operations: {len(proc.processing_history)}")
    
    # ========== FINAL SUMMARY ==========
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    summary = pipeline.get_summary()
    
    print(f"\nPipeline ID: {summary['pipeline_id']}")
    print(f"Total Steps: {summary['pipeline_steps']}")
    print(f"Final Data Shape: {summary['data_shape']}")
    
    print(f"\nComponents Initialized:")
    for component, initialized in summary['components_initialized'].items():
        status = "✓" if initialized else "✗"
        print(f"  {status} {component.capitalize()}")
    
    print(f"\nPending Operations:")
    for op in summary['pending_operations']:
        print(f"  - {op}")
    
    print("\n" + "="*80)
    print("PROJECT 4 ENHANCEMENTS DEMONSTRATED:")
    print("="*80)
    print("✓ Enhanced base classes with UUID and temporal tracking")
    print("✓ Registry pattern for sources and processors")
    print("✓ DataQualityStandards business logic layer")
    print("✓ Factory methods for component creation")
    print("✓ Composition over inheritance (HAS-A relationships)")
    print("✓ Advanced analytics (schedule, lineage, quality reports)")
    print("✓ Performance monitoring across all components")
    print("✓ File versioning and checksums")
    print("✓ Complete pipeline orchestration")
    print("✓ Polymorphic behavior across sources and processors")
    print("✓ Export functionality for audit trails")
    print("="*80)
    
    print("\n✓ DEMONSTRATION COMPLETE!")
    print(f"✓ Output saved to: {output_dir.absolute()}")
    
    print("\n[CLEANUP] Removing temporary files...") # cleanup
    temp_csv.unlink()
    
    import shutil
    shutil.rmtree(output_dir)
    
    print("✓ Cleanup complete")
    
    # Clear registries
    DataSource.clear_registry()
    DataProcessor.clear_registry()
    
    print("\n" + "="*80)
    print("Thank you for exploring the Crime Research Data Pipeline!")
    print("="*80 + "\n")
    
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
