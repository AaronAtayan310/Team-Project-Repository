## Crime Research Data Pipeline - Advanced OOP Architecture Diagram
This file lists the crime research data pipeline's inheritance hierarchies 
and their rationales, the reasoning behind why specific methods in the 
project are polymorphic, and the reasoning behind major compposition vs 
inheritance decisions in the project design.  

## Complete System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          ABSTRACT BASE LAYER                             │
│                     (Defines Interface Contracts)                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────────────────────┐  ┌─────────────────────────────┐     │
│  │ AbstractDataProcessor          │  │ AbstractDataSource          │     │
│  ├────────────────────────────────┤  ├─────────────────────────────┤     │
│  │ + frame (DataFrame)            │  │ + metadata (Dict)           │     │
│  │ + processing_history (List)    │  │                             │     │
│  │ + verbose (bool)               │  │ @abstractmethod             │     │
│  │                                │  │ + load() → DataFrame        │     │
│  │ @abstractmethod                │  │ + validate_source() → bool  │     │
│  │ + process() → self             │  │                             │     │
│  │ + validate() → bool            │  └─────────────────────────────┘     │
│  └────────────────────────────────┘            ▲                         │
│             ▲                                  │ inherits                │
│             │ inherits                         │                         │
└─────────────┼──────────────────────────────────┼─────────────────────────┘
              │                                  │
┌─────────────┼──────────────────────────────────┼─────────────────────────┐
│             │         CONCRETE IMPLEMENTATION LAYER                      │
│             │              (Inheritance & Polymorphism)                  │
├─────────────┴──────────────────────────────────┴─────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────────┐       │
│  │ NewDataAnalysis │  │NewDataCleaner│  │ NewDataTransformation  │       │
│  ├─────────────────┤  ├──────────────┤  ├────────────────────────┤       │
│  │ + described     │  │ + original_  │  │ + scalers              │       │
│  │ + models        │  │   shape      │  │                        │       │
│  │                 │  │ + cleaning_  │  │ process()              │       │
│  │ process()       │  │   history    │  │ → generate_features()  │       │
│  │ → describe()    │  │              │  │ scale_features()       │       │
│  │                 │  │ process()    │  │ → StandardScaler       │       │
│  │ crime_rate_     │  │ → mean/median│  │                        │       │
│  │ by_year()       │  │ /drop rows   │  │ validate()             │       │
│  │ top_crime_      │  │ normalize_   │  │ → no inf values        │       │
│  │ types()         │  │ text()       │  │                        │       │
│  └─────────────────┘  └──────────────┘  └────────────────────────┘       │
│                                                                          │
│  ┌─────────────────┐    ┌──────────────┐    ┌─────────────────────┐      │
│  │  APIDataSource  │    │ CSVDataSource│    │ DatabaseDataSource  │      │
│  ├─────────────────┤    ├──────────────┤    ├─────────────────────┤      │
│  │ + url, params   │    │ + filepath   │    │ + conn_string       │      │
│  │                 │    │              │    │ + query             │      │
│  │ load()          │    │ load()       │    │ load()              │      │
│  │ → requests.get  │    │ → pd.read_csv│    │ → SQLAlchemy        │      │
│  │ → JSON→DataFrame│    │              │    │ → pd.read_sql       │      │
│  │ validate_       │    │ validate_    │    │ validate_           │      │
│  │ source()        │    │ source()     │    │ source()            │      │
│  │ → http(s) URL   │    │ → file exists│    │ → conn+query valid  │      │
│  └─────────────────┘    └──────────────┘    └─────────────────────┘      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
                                 │
                                 │ used by
                                 ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                       COMPOSITION LAYER                                   │
│                   (System Coordination)                                   │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                      DataPipeline                               │      │
│  ├─────────────────────────────────────────────────────────────────┤      │
│  │ + ingestion: NewDataIngestion                                   │      │
│  │ + storage: NewDataStorageUtils                                  │      │
│  │ + cleaner/transformer/analyzer (dynamic)                        │      │
│  │ + data (current DataFrame)                                      │      │
│  │ + pipeline_history                                              │      │
│  │                                                                 │      │
│  │ + load_data(source) → polymorphic load                          │      │
│  │ + clean()/transform()/analyze() → composition                   │      │
│  │ + save_results() → storage utils                                │      │
│  │ + run_full_pipeline() → end-to-end                              │      │
│  │ + get_summary() → pipeline metrics                              │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                │                                          │
│                                │ orchestrates                             │
│                                ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                    NewDataIngestion                             │      │
│  ├─────────────────────────────────────────────────────────────────┤      │
│  │ + default_timeout                                               │      │
│  │ + track_sources                                                 │      │
│  │ + data_sources[], loaded_data{}                                 │      │
│  │                                                                 │      │
│  │ + load_from_source(source) → polymorphic                        │      │
│  │ + load_csv() → CSVDataSource                                    │      │
│  │ + fetch_api_data()                                              │      │
│  └─────────────────────────────────────────────────────────────────┘      │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐      │
│  │                  NewDataStorageUtils                            │      │
│  ├─────────────────────────────────────────────────────────────────┤      │
│  │ + base_output_dir                                               │      │
│  │ + logger                                                        │      │
│  │                                                                 │      │
│  │ + save_to_csv() → timestamped filenames                         │      │
│  │ + serialize_model() → pickle + JSON metadata                    │      │
│  │ + generate_timestamped_filename()                               │      │
│  └─────────────────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────────────┘
```

## Key Relationships

### INHERITANCE (is-a)
- NewDataAnalysis **IS-A** AbstractDataProcessor
- NewDataCleaner **IS-A** AbstractDataProcessor  
- NewDataTransformation **IS-A** AbstractDataProcessor
- APIDataSource **IS-A** AbstractDataSource
- CSVDataSource **IS-A** AbstractDataSource
- DatabaseDataSource **IS-A** AbstractDataSource

### COMPOSITION (has-a / has-many)
- DataPipeline **HAS-A** NewDataIngestion (data loading)
- DataPipeline **HAS-A** NewDataStorageUtils (file I/O)
- DataPipeline **HAS-MANY** processors (cleaner/transformer/analyzer - dynamic)
- NewDataIngestion **HAS-MANY** data_sources[] + loaded_data{}
- AbstractDataProcessor **HAS-A** frame (DataFrame) + processing_history[]

### POLYMORPHISM (same method, different behavior)
- All processors implement `process()` differently:
  - NewDataAnalysis: statistical description + crime analytics
  - NewDataCleaner: missing value strategies + text normalization
  - NewDataTransformation: feature generation + StandardScaler

- All sources implement `load()` differently:
  - APIDataSource: requests.get() → JSON → DataFrame
  - CSVDataSource: pd.read_csv() → file validation
  - DatabaseDataSource: SQLAlchemy → pd.read_sql()

## Data Flow Example

```
1. User creates DataPipeline
   └─→ Pipeline initializes with NewDataIngestion + NewDataStorageUtils

2. User loads data polymorphically
   └─→ pipeline.load_data(CSVDataSource("crime.csv"))
   └─→ NewDataIngestion.load_from_source() → CSVDataSource.load()
   └─→ DataFrame cached in pipeline.data

3. Pipeline coordinates processing via composition
   └─→ pipeline.clean(strategy='mean') → NewDataCleaner created
   └─→ pipeline.transform(['crime_count']) → NewDataTransformation
   └─→ pipeline.analyze() → NewDataAnalysis

4. Results saved through composition
   └─→ pipeline.save_results("output.csv", use_timestamp=True)
   └─→ NewDataStorageUtils.save_to_csv() → timestamped filename

5. User checks pipeline status
   └─→ pipeline.get_summary()
   └─→ Returns metrics across all components (inheritance polymorphism)
```

## Why This Architecture Works

### Abstract Layer Benefits
✓ Enforces consistent DataFrame processing/loading interfaces
✓ Cannot instantiate incomplete processor/source classes
✓ Clear contracts for future extensions (new sources/processors)
✓ Type safety through Python ABC module

### Concrete Layer Benefits
✓ Specialized crime analytics (crime_rate_by_year, top_crime_types)
✓ Polymorphic data loading (CSV/API/DB interchangeable)
✓ Domain-specific validation (URL format, file existence, SQL syntax)
✓ Easy extension with new processor/source types

### Composition Layer Benefits
✓ Fluent method chaining (pipeline.load().clean().transform().save())
✓ Dependency injection for testability
✓ Dynamic processor creation (no fixed processor hierarchy)
✓ Centralized pipeline orchestration + logging

## Design Pattern Summary

| Pattern                  | Where Used                                | Why                                 |
|--------------------------|-------------------------------------------|-------------------------------------|
| **Abstract Base Class**  | AbstractDataProcessor, AbstractDataSource | Enforce interface contracts         |
| **Inheritance**          | Processor/Source hierarchies              | Code reuse + polymorphism           |
| **Polymorphism**         | process()/load() methods                  | Source/processor-specific behavior  |
| **Composition**          | DataPipeline, NewDataIngestion            | Flexible component coordination     |
| **Fluent Interface**     | Method chaining (return self)             | Readable pipeline syntax            |
| **Dependency Injection** | DataPipeline constructor                  | Testability + flexibility           |
| **Template Method**      | Abstract process()/validate()             | Define pipeline algorithm structure |

This architecture demonstrates professional OOP design principles applied to real crime research pipelines, providing a solid foundation for an application that has real relevance to Information Science.
