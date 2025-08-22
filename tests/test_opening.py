import pytest
import xarray as xr
import numpy as np
from pathlib import Path

# Path to the NetCDF file
DATA_PATH = Path(__file__).parent / "../magnetics_diagnostic_analysis/data/train.nc"


@pytest.fixture(scope="module")
def dataset():
    if not DATA_PATH.exists():
        pytest.fail(f"File {DATA_PATH} does not exist.")
    with xr.open_dataset(DATA_PATH) as ds:
        yield ds.load()


def test_dataset_is_opened(dataset):
    """Test that the dataset is properly opened and is an xarray.Dataset object"""
    assert isinstance(dataset, xr.Dataset), "The file is not an xarray Dataset."


def test_dataset_dimensions(dataset):
    """Test that the dataset contains at least the basic expected dimensions"""
    expected_dims = {"time", "shot_id"}
    actual_dims = set(dataset.dims)
    missing = expected_dims - actual_dims
    assert not missing, f"Missing dimensions: {missing}"


def test_dataset_has_data_vars(dataset):
    """Check that the dataset contains at least one data variable"""
    assert len(dataset.data_vars) > 0, "The dataset contains no data variables."


def test_time_coordinate_is_monotonic(dataset):
    """Check that the time coordinate is monotonically increasing"""
    time = dataset.coords.get("time", None)
    assert time is not None, "'time' coordinate is missing."
    assert np.all(np.diff(time.values) >= 0), "'time' coordinate is not monotonically increasing."


def test_shot_id_coordinate_exists(dataset):
    """Check that the 'shot_id' coordinate exists and contains valid values"""
    assert "shot_id" in dataset.coords, "'shot_id' coordinate is missing."
    assert dataset.coords["shot_id"].size > 0, "'shot_id' coordinate is empty."

@pytest.mark.skip(reason="Not for the moment")
def test_values_are_finite(dataset):
    """Ensure that all data values are finite (no NaNs or infinite values)"""
    for var_name in dataset.data_vars:
        data = dataset[var_name].values
        assert np.all(np.isfinite(data)), f"Variable '{var_name}' contains NaN or infinite values."


def test_specific_variable_presence(dataset):
    """Test that the expected magnetic field variable is present"""
    expected_var = "b_field_pol_probe_ccbv_field"
    assert expected_var in dataset.variables, f"Expected variable '{expected_var}' not found."


def test_dataset_attributes(dataset):
    """Check that the dataset has global attributes"""
    assert isinstance(dataset.attrs, dict), "Global attributes are not a dictionary."
