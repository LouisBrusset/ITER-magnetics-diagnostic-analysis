import numpy as np
import xarray as xr
import pandas as pd

import json
from pathlib import Path

from magnetics_diagnostic_analysis.project_mscred.setting_mscred import config

def print_dataset_info(ds: xr.Dataset) -> None:
    print(f"{'\nVariable':<40} {'Shape':<20} {'Dims':<55} {'NaN Count':<10}")
    for var in ds.data_vars:
        shape = ds[var].shape
        dims = ds[var].dims
        nan_count = ds[var].isnull().sum().values
        print(f"{var:<40} {str(shape):<20} {str(dims):<55} {nan_count:<10}")
    return None


def filter_xr_dataset_channels(data_train: xr.Dataset, var_channel_df: pd.DataFrame) -> xr.Dataset:
    """
    Filters the channels of the data_train variables according to var_channel_df.
    Removes variables not listed in var_channel_df.

    Args:
        data_train (xr.Dataset): Dataset containing variables with or without the “channel” dimension.
        var_channel_df (pd.DataFrame): DataFrame with columns ['variable', 'channel'].

    Returns:
        xr.Dataset: Filtered dataset.
    """
    variables_to_keep = set(var_channel_df["variable"])
    
    filtered_data = {}
    for variable in variables_to_keep:
        da = data_train[variable]
        da_dims = da.dims
        coord = [dim for dim in da_dims if 'channel' in dim]
        if not coord:
            da_filtered = da
        else:
            coord_names = da[coord[0]].values
            coord_names_to_keep = var_channel_df.loc[var_channel_df["variable"] == variable, "channel"].tolist()
            channel_indices = np.where(np.isin(coord_names, coord_names_to_keep))[0]
            da_filtered = da.isel({coord[0]: channel_indices})
        filtered_data[variable] = da_filtered

    final = xr.Dataset(filtered_data)
    final["shot_index"] = data_train["shot_index"]

    return final

def impute_to_zero(data: xr.Dataset) -> xr.Dataset:
    """
    Imputes missing values in the dataset by replacing them with zero.
    We loop over the variables, so that we do not modify the coordinates and the dataset structure (attrs for instance)
    """
    result = data.copy()
    for var in result.data_vars:
        result[var] = result[var].fillna(0)
    return result


def clean_data(vars: list[str], group:str = "magnetics", suffix: str = "mscred", verbose: bool = False) -> None:
    """
    Clean the dataset by removing NaN values and selecting good variables.

    Parameters:
    vars (list[str]): The list of variable names to keep.
    group (str): The group name for the dataset.
    suffix (str): The suffix for the dataset file.
    verbose (bool): If True, print dataset information before and after cleaning.

    Returns:
    None : We directly saved the cleaned dataset to a NetCDF file.
    """
    # Load the dataset
    path = config.DIR_RAW_DATA / config.RAW_DATA_FILE_NAME
    with xr.open_dataset(path) as f:
        #subset = train.sel(shot_id=shots[])
        ds = f.load()
    if verbose:
        print("Before cleaning:")
        print_dataset_info(ds)

    # Load the variables and channels to keep
    #path = Path(__file__).absolute().parent.parent.parent.parent / f"notebooks/result_files/all_shots_{group}" / f"result_lists_{group}.json"
    #with open(path, "r") as f:
    #    data_var = json.load(f)
    #good_vars_name = data_var["good_vars_ids"]
    var_channel_df = pd.DataFrame(vars, columns=["full_name"])
    var_channel_df[["variable", "channel"]] = var_channel_df["full_name"].str.split("::", expand=True)


    # Select only the good variables
    ds_filtered = filter_xr_dataset_channels(ds, var_channel_df)
    if verbose:
        print("After channel filtering:")
        print_dataset_info(ds_filtered)

    # Drop rows with NaN values in any of the selected variables
    ds_cleaned = impute_to_zero(ds_filtered)
    if verbose:
        print("After dropping NaNs:")
        print_dataset_info(ds_cleaned)

    ### Save the cleaned dataset
    path_out = config.DIR_PREPROCESSED_DATA / f"data_{group}_{suffix}_cleaned.nc"
    ds_cleaned.to_netcdf(path_out)

    return ds_cleaned


if __name__ == "__main__":
    clean_data(verbose=True)  # Default is magnetics and mscred