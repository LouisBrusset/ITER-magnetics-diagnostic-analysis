# Data Downloading Module

This module provides comprehensive utilities for downloading, loading, and preprocessing experimental data from the MAST (Mega Amp Spherical Tokamak) experiment database.

The scientific data is accessed directly via the FAIR-MAST database API, find the link here: https://mastapp.site/

## Table of Contents

1. [Presentation and Purpose](#presentation-and-purpose)
2. [File Enumeration and Purpose](#file-enumeration-and-purpose)
3. [Functions by File](#functions-by-file)
   - [data_downloading.py](#-data_downloadingpy)
   - [steady_state_filtering.py](#-steady_state_filteringpy)
   - [data_washing.py](#-data_washingpy)
4. [Usage Examples](#usage-examples)
5. [Data Sources and Formats](#data-sources-and-formats)

## Presentation and Purpose

This module serves as the **data foundation** for all machine learning projects in the package. It provides:

- **MAST Database Integration**: Direct access to tokamak experimental data
- **Robust Data Pipeline**: From raw database to ML-ready datasets
- **Quality Control**: Filtering and cleaning utilities
- **Steady State Detection**: Physics-informed filtering for tokamak operations

The module is designed to be the **single entry point** for all experimental data needs across MSCRED, VAE, and SCINet projects.

## File Enumeration and Purpose

### 📡 **`data_downloading.py`**
Core data acquisition and dataset building functions
- Shot listing, data retrieval from MAST database
- Dataset construction with train/test splitting
- Retry mechanisms for robust data downloading

### ⚖️ **`steady_state_filtering.py`**
Physics-informed filtering for tokamak steady state detection
- Plasma current analysis and filtering
- Steady state period identification
- Connected component analysis for temporal continuity

### 🧹 **`data_washing.py`**
Data cleaning and preprocessing utilities
- NaN handling through selection and imputation
- Channel filtering and quality assessment
- Dataset information and validation tools

## Functions by File

### 📡 `data_downloading.py`

#### `shot_list(campaign, quality)`
Retrieves list of shot IDs from MAST database
- **campaign**: Campaign identifier (e.g., "M9")
- **quality**: Filter for quality shots only
- **Returns**: List of shot IDs

#### `to_dask(shot, group, level)`
Downloads single shot data as xarray Dataset
- **shot**: Shot number to download
- **group**: Data group ("magnetics", "efit", etc.)
- **level**: Data processing level (default: 2)

#### `retry_to_dask(shot_id, group, retries, delay)`
Robust data downloading with retry mechanism
- **retries**: Number of retry attempts
- **delay**: Delay between retries in seconds

#### `build_level_2_data_all_shots(shots, groups, steady_state, verbose)`
Builds concatenated dataset from multiple shots
- Combines all shots into single xarray Dataset
- Efficient for batch processing

#### `load_data(shots, groups, steady_state, verbose)`
High-level data loading function
- Simplified interface for quick data access

### ⚖️ `steady_state_filtering.py`

#### `ip_filter(ip, filter, min_current)`
Main plasma current filtering function
- **ip**: Plasma current time series
- **filter**: Filter type ("default", "bidirectional", "unidirectional")
- **min_current**: Minimum current threshold (default: 4e4 A)
- **Returns**: Boolean mask for steady state periods

#### `_keep_largest_connected_component(mask)`
Internal function to select largest continuous time segment
- Ensures temporal continuity in filtered data

#### `_filter_low_current_regions(ip, mask, min_current)`
Internal function to remove low current periods
- Physics-based filtering for meaningful tokamak operation

### 🧹 `data_washing.py`

#### `print_dataset_info(ds)`
Displays comprehensive dataset information
- Variable names, dimensions, data types
- NaN statistics and data quality metrics

#### `filter_xr_dataset_channels(data_train, var_channel_df)`
Filters dataset channels based on quality criteria
- **var_channel_df**: DataFrame with channel quality information
- **Returns**: Filtered xarray Dataset

#### `impute_to_zero(data)`
Imputes NaN values to zero
- Simple imputation strategy for missing data
- **Returns**: Dataset with NaNs replaced

#### `clean_data(vars, group, suffix, verbose)`
High-level data cleaning pipeline
- **vars**: List of variables to clean
- **group**: Diagnostic group to process
- **suffix**: Output file identifier

## Usage Examples

### Basic Data Loading
```python
from magnetics_diagnostic_analysis.data_downloading import data_downloading

# Get quality shots from M9 campaign
shots = data_downloading.shot_list("M9", quality=True)

# Load magnetics data for analysis
data_downloading.load_data(
    shots=shots[:10],
    groups=["magnetics"],
    steady_state=True,
    verbose=True
)
```

### Data Cleaning Pipeline
```python
from magnetics_diagnostic_analysis.data_downloading import data_washing

# Clean and preprocess data
data_washing.clean_data(
    vars=["ip"],
    group="magnetics",
    suffix="mscred",
    verbose=True
)
```

### Steady State Filtering
```python
from magnetics_diagnostic_analysis.data_downloading import steady_state_filtering
import numpy as np

# Apply plasma current filtering
ip_data = np.array([...])  # Your plasma current data
steady_mask = steady_state_filtering.ip_filter(
    ip=ip_data,
    filter="bidirectional",
    min_current=5e4
)
```

## Data Sources and Formats

### **MAST Database**
- **URL**: https://mastapp.site/
- **Access**: Direct API integration via dask
- **Campaigns**: M8, M9 (primary focus)

### **Data Formats**
- **Input**: NetCDF4 via xarray from MAST database
- **Output**: NetCDF4 (.nc) files for local storage
- **Processing**: Zarr-compatible for efficient access

### **Diagnostic Groups**
- **magnetics**: Magnetic diagnostic measurements
- **efit**: Equilibrium reconstruction data
- **summary**: Shot summary parameters

### **Quality Control**
- Automatic retry mechanisms for network failures
- NaN detection and handling strategies
- Physics-informed filtering for data validity
- Steady state detection based on plasma current analysis