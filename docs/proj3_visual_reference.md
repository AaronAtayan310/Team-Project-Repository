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
