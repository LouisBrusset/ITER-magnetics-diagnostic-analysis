# Data Loading Module

This module provides utilities for downloading and loading data from the MAST experiment database.

## Table of Contents

1. [Files](#-files)
2. [Main Functions](#-main-functions)
   - [`shot_list`](#shot_listcampaign-qualitynone)
   - [`load_data`](#load_datafile_path-suffix-train_test_rate-shots-groups-permanent_state)
   - [`build_level_2_data_per_shot`](#build_level_2_data_per_shotshots-groups-permanent_statefalse)
   - [`ip_filter`](#ip_filterip-filterdefault-min_currentnone)
3. [Filtering by Plasma Current Smoothness](#-filtering-by-plasma-current-smoothness)
4. [Data Sources](#-data-sources)
5. [Quick Example](#-quick-example)
6. [Notes](#-notes)

## 📁 Files

- **`data_downloading.py`** - Core functions for downloading and processing MAST data
- **`permanent_state_filtering.py`** - Filtering utilities for tokamak permanent state detection
- **`opening_test.ipynb`** - Example notebook for testing data loading functionality

## 🔧 Main Functions

### `shot_list(campaign="", quality=None)`
Get a list of shot IDs from the MAST database.

```python
from magnetics_diagnostic_analysis.data_loading import shot_list

# Get all shots from M9 campaign
shots = shot_list("M9")

# Get only quality shots
good_shots = shot_list("M9", quality=True)
```

### `load_data(file_path, suffix, train_test_rate, shots, groups, permanent_state)`
Load and cache data for training and testing.

```python
from magnetics_diagnostic_analysis.data_loading import load_data

load_data(
    file_path="data/",
    suffix="_mscred",
    train_test_rate=0.8,
    shots=shots[:100],
    groups=["magnetics"],
    permanent_state=True,
    verbose=True
)
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