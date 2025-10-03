# Data Downloading Module

This module provides utilities for downloading, loading, and preprocessing data from the MAST experiment database.

## Table of Contents

1. [Files](#-files)
2. [Main Functions](#-main-functions)
3. [Data Sources](#-data-sources)
4. [Quick Example](#-quick-example)
5. [Notes](#-notes)

## 📁 Files

- **`data_downloading.py`** - Core functions for downloading and processing MAST data
- **`steady_state_filtering.py`** - Filtering utilities for tokamak steady state detection based on the plasma current signal
- **`data_washing.py`** - Data cleaning and preprocessing utilities, mainly removing NaNs by selecting or imputing

## 🔧 Main Functions

### Data Downloading and Processing
Functions for retrieving and processing experimental data from the MAST database.

```python
from magnetics_diagnostic_analysis.data_downloading import data_downloading

# Example usage (specific API depends on implementation)
data = data_downloading.load_data()
```

### Steady State Filtering
Utilities for detecting and filtering tokamak steady state periods.

```python
from magnetics_diagnostic_analysis.data_downloading import steady_state_filtering

# Apply steady state filtering
filtered_data = steady_state_filtering.filter_steady_state(data)
```

### Data Washing
Data cleaning and preprocessing functions.

```python
from magnetics_diagnostic_analysis.data_downloading import data_washing

# Clean and preprocess data
clean_data = data_washing.clean_data(raw_data)
```


### `build_level_2_data_per_shot(shots, groups, permanent_state=False)`
Build dataset with one file per shot.

```python
dataset = build_level_2_data_per_shot(
    shots=[30001, 30002, 30003],
    groups=["magnetics", "efit"],
    permanent_state=True
)
```

### `ip_filter(ip, filter='default', min_current=None)`
Filter plasma current data to identify permanent state periods.

```python
from magnetics_diagnostic_analysis.data_loading import ip_filter

# Apply filtering to current data
mask = ip_filter(current_data, filter='bidirectional', min_current=4e4)
```

## 📊 Filtering by Plasma Current Smoothness

The `permanent_state_filtering.py` file provides two filtering approaches:

1. **`scipy.lfilter`**: Unidirectional lowpass filter
   - ⚠️ Introduces a delay of `(ntaps-1)/2` samples
   - ✅ Faster computation
   
2. **`scipy.filtfilt`**: Bidirectional filter
   - ✅ No delay (zero-phase)
   - ⚠️ Twice the computational time

## 📊 Data Sources

- **MAST Database**: https://mastapp.site/
- **Data Format**: NetCDF4 via xarray
- **Storage**: Zarr format for efficient access

## 🚀 Quick Example

```python
from magnetics_diagnostic_analysis.data_loading import shot_list, load_data

# Get shots from M9 campaign
shots = shot_list("M9", quality=True)

# Load magnetics data for first 10 shots
load_data(
    file_path="./data",
    suffix="_example",
    train_test_rate=0.8,
    shots=shots[:10],
    groups=["magnetics"],
    permanent_state=True,
    verbose=True
)

print("Data loaded successfully!")
print(f"Training data: train_example.nc")
print(f"Test data: test_example.nc")
```

## ⚠️ Notes

- Data loading requires internet connection to access MAST database
- Large datasets are cached locally to avoid re-downloading
- Use `permanent_state=True` for steady-state analysis
- The `ip_filter` function is crucial for identifying relevant time periods in tokamak data