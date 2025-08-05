import numpy as np
import pandas as pd
import xarray as xr
import random as rd
import time

import pathlib
import tqdm
import requests

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from permanent_state_filtering import ip_filter


def shot_list(campaign: str = "", quality: bool = None) -> list[int]:
    """
    Return a list of shot IDs for a given campaign.

    Parameters
    campaign: Campaign name to filter shots. If None, return all campaigns.
    quality: If True, return only shots with 'cpf_useful' label. If False, return shots with 'cpf_abort' label.
    
    Returns
    List of shot IDs.
    """
    URL = "https://mastapp.site"
    if campaign == "":
        shots_df = pd.read_parquet(f'{URL}/parquet/level2/shots')
    else:
        shots_df = pd.read_parquet(f'{URL}/parquet/level2/shots?filters=campaign$eq:{campaign}')
    
    shots_df['shot_label'] = (shots_df['cpf_useful'] == 1).astype(int)
    if quality is None:
        return shots_df['shot_id'].tolist()
    elif quality == True:
        return shots_df.loc[shots_df['shot_label'] == 1, 'shot_id'].tolist()
    else:
        return shots_df.loc[shots_df['shot_label'] == 0, 'shot_id'].tolist()


def to_dask(shot: int, group: str, level: int = 2) -> xr.Dataset:
    """
    Return a Dataset from the MAST Zarr store.

    Parameters
    shot: Shot ID to retrieve data for.
    group: Diagnostic group to retrieve data from.
    level: Data level to retrieve (default is 2).
    """
    return xr.open_zarr(
        f"https://s3.echo.stfc.ac.uk/mast/level{level}/shots/{shot}.zarr",
        group=group
    )


def retry_to_dask(shot_id: int, group: str, retries: int = 5, delay: int = 1):
    """
    Retry loading a shot's data as a Dask Dataset with exponential backoff.

    Parameters
    shot_id: Shot ID to retrieve data for.
    group: Diagnostic group to retrieve data from.
    retries: Number of retry attempts (default is 3).
    delay: Delay in seconds between retries (default is 5).

    Returns
    xr.Dataset
        The Dask Dataset for the specified shot and group.
    or Error
    """
    for attempt in range(retries):
        try:
            return to_dask(shot_id, group)
        except Exception as e:
            if attempt < retries - 1:
                print(f"Retrying connection to {shot_id} in group {group} (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
            else:
                raise e
            

def build_level_2_data_per_shot(shots: list[int], groups: list[str], permanent_state: bool = False, verbose: bool = False) -> xr.Dataset:
    """
    Warning: This function is deprecated and will be removed in future versions.
    The aim was to build a dataset with shot_id as a dimension, but aligning problems have occurred.
    Use `build_level_2_data_all_shot` instead.
    ==============

    
    Retrieve specified groups of diagnostics from shots in the M9 campaign during permanent state or not.
    
    Parameters
    shots: List of shot IDs to retrieve data for.
    groups: List of diagnostic groups to retrieve. If None, all groups are retrieved.
    permanent_state: If True, only retrieve shots during the permanent state phase of the campaign.

    Return
    An xarray Dataset containing the requested diagnostic data.
    """

    dataset = []

    for shot in tqdm.tqdm(shots, desc="Loading shots", total=len(shots)):
        try:
            summary = retry_to_dask(shot, "summary")
        except (IndexError, KeyError):
            print(f"Issue on 'summary' for shot {shot}")
            continue
        ip = summary['ip']
        time_ip = summary['time']

        if permanent_state:
            mask, _, _ = ip_filter(ip.values, filter='default', min_current=4.e4)
            time_ref = time_ip[mask]
        else:
            time_ref = time_ip

        time_dim_name = f"time_{shot}"      # Create a unique name for a given shot
        shot_signals = []
        if verbose:
            print("time_dim_name = ", time_dim_name, "\n")

        for group in groups:
            if verbose:
                print("\n", f"Loading group {group} for shot {shot}...", "\n")
            try:
                data = retry_to_dask(shot, group)
            except (IndexError, KeyError):
                print(f"Group {group} not found for shot {shot}. Skipping.")
                continue

            if group == "summary":
                # If the group is summary, we do not need the 'ip' variable
                data = data.drop_vars("ip", errors="ignore")

            interpolated_vars = {}
            other_time_coords = set()
            
            for var_name, da in data.data_vars.items():
                if verbose:
                    print(f"Processing variable: {var_name}")
                # Find the time dimension (could be "time", "time_saddle", etc.)
                time_dim = next((d for d in da.dims if d.startswith("time")), None)
                
                if time_dim is not None:
                    da_interp = da.interp({time_dim: time_ref})
                    other_time_coords.add(time_dim)
                else:
                    da_interp = da

                da_interp.attrs |= {"group": group}
                interpolated_vars[var_name] = da_interp

            cleaned = xr.Dataset(interpolated_vars)

            # Drop unnecessary time coordinates
            for coord in other_time_coords:
                if coord in cleaned.coords:
                    cleaned = cleaned.drop_vars(coord)
            # Final step: rename "time" to a unique name for this shot
            cleaned = cleaned.swap_dims({'time': time_dim_name})
            cleaned = cleaned.assign_coords({time_dim_name: time_ref.values})
            cleaned = cleaned.transpose(time_dim_name, ...)
            if verbose:
                print(cleaned)

            shot_signals.append(cleaned)
            
        if not shot_signals:
            continue

        shot_data = xr.merge(shot_signals, combine_attrs="drop_conflicts")
        shot_data = shot_data.expand_dims("shot_id")
        shot_data = shot_data.assign_coords(shot_id=("shot_id", [shot]))

        dataset.append(shot_data)

    if not dataset:
        raise ValueError("No shot data found.")

    final = xr.concat(dataset, dim="shot_id", coords="minimal")
    print("dataset ok")
    
    return final



def build_level_2_data_all_shots(shots: list[int], groups: list[str], permanent_state: bool = False, verbose: bool = False) -> xr.Dataset:
    """
    Retrieve specified groups of diagnostics from shots in the M9 campaign during permanent state or not.
    
    Parameters
    shots: List of shot IDs to retrieve data for.
    groups: List of diagnostic groups to retrieve. If None, all groups are retrieved.
    permanent_state: If True, only retrieve shots during the permanent state phase of the campaign.

    Return
    An xarray Concatenated dataset containing the requested diagnostic data, aligned on 'time' with 'shot_index' and 'shot_id'.
    """
    
    dataset = []

    for shot_index, shot_id in tqdm.tqdm(enumerate(shots), desc="Loading shots", total=len(shots)):
        try:
            ref = retry_to_dask(shot_id, "summary")
        except (IndexError, KeyError):
            print(f"Issue on 'summary' for shot {shot_id}")
            continue
        ip_ref = ref['ip']
        time = ref.time

        if permanent_state:
            mask, _, _ = ip_filter(ip_ref.values, filter='default', min_current=4.e4)
            time_ref = time[mask]
        else:
            time_ref = time
        if len(time_ref) == 0:
            print(f"Skipping shot {shot_id} due to empty time_ref.")
            continue

        signal = []

        for group in groups:
            if verbose:
                print("\n", f"Loading group {group} for shot {shot_id}...", "\n")
            try:
                data = retry_to_dask(shot_id, group).interp({"time": time_ref})
            except (IndexError, KeyError):
                print(f"\nGroup {group} not found for shot {shot_id}. Skipping.")
                continue

            if group == "summary":
                # If the group is summary, we do not need the 'ip' variable. Else issue with ip alignment in magnetics.
                data = data.drop_vars("ip", errors="ignore")
            
            other_times = set()
            for var in data.data_vars:
                if verbose:
                    print(f"Processing variable: {var}")
                time_dim = next((dim for dim in data[var].dims if dim.startswith('time')), 'time')

                if time_dim != "time":
                    other_times.add(time_dim)
                data[var] = data[var].interp({time_dim: time_ref})               
                data[var] = data[var].transpose("time", ...)
                data[var].attrs |= {"group": group}
                #data[var].attrs |= {"timestamp": requests.get(f'https://mastapp.site/json/shots/{shot}').json()["timestamp"]}
                # Timestamp goes away with concat, so we don't add it here

            data = data.drop_vars(other_times)
            signal.append(data)

        signal = xr.merge(signal, combine_attrs="drop_conflicts")
        signal["shot_index"] = "time", shot_index * np.ones(len(time_ref), int)
        dataset.append(signal)

    final = xr.concat(dataset, "time", join="override", coords="minimal", combine_attrs="drop_conflicts")           # coords="minimal" deletes all coords that are not in all datasets
    print("dataset ok")

    return final



def load_data(file_path: str, suffix: str, train_test_rate: float, shots: list[int], groups: list[str], permanent_state: bool, random_seed: int = 42, verbose: bool = False) -> None:
    """
    Load data from cache or build it if not available.

    Parameters
    train_test_rate: Proportion of shots to use for training (1 - test).
    shots: List of shot IDs to retrieve data for.
    groups: List of diagnostic groups to retrieve data from.
    permanent_state: If True, only retrieve shots during the permanent state phase of the campaign.
    random_seed: Seed for random number generator to shuffle shots.
    
    Return
    An xarray Dataset containing the requested diagnostic data.
    """
    assert 0.05 < train_test_rate < 0.95, "train_test_rate must be between 0.05 and 0.95"

    path = pathlib.Path().absolute() / file_path
    path.mkdir(exist_ok=True)
    filename_train = path / f"train{suffix}.nc"
    filename_test = path / f"test{suffix}.nc"

    try:
        with open(filename_train, "rb") as ftrain, open(filename_test, "rb") as ftest:
            print("Files already exist!")

    except FileNotFoundError:
        rng = np.random.default_rng(random_seed)
        rng.shuffle(shots)
        
        split_id = int(len(shots) * (1-train_test_rate))
        split_ids = {
            "train": shots[:split_id],
            "test": shots[split_id:],
        }
        #dataset = {mode: build_level_2_data_all_shots(shots=shot_ids, groups=groups, permanent_state=permanent_state) for mode, shot_ids in split_ids.items()}
        dataset = {mode: build_level_2_data_all_shots(
            shot_ids, 
            groups=groups, 
            permanent_state=permanent_state,
            verbose=verbose
            ) for mode, shot_ids in split_ids.items()}
        print("Saving to netCDF...")
        dataset["train"].to_netcdf(filename_train)
        dataset["test"].to_netcdf(filename_test)
        print("netCDF ok")

    return None



if __name__ == "__main__":

    n_samples = 12       # Number of shots to load
    random_seed = 42
    campaign_number = ""

    shots = shot_list(campaign=campaign_number, quality=True)
    rd.seed(random_seed)
    rd.shuffle(shots)
    shots = shots[:n_samples]
    print("Chosen shots: \n", shots)
    print("Type of shots: ", type(shots))


    #shots = [15585, 15212, 15010, 14998, 30410, 30418, 30420]

    groups = ["summary", "magnetics", "spectrometer_visible", "pf_active"]
    permanent_state = False
    train_test_rate = 0.3
    file_path = "src/magnetics_diagnostic_analysis/data"
    suffix = "_test"

    load_data(
        file_path=file_path, 
        suffix=suffix,
        train_test_rate=train_test_rate, 
        shots=shots, groups=groups, 
        permanent_state=permanent_state, 
        random_seed=random_seed,
        verbose=False
        )
    
    print("Data loading completed.\n")
    
    path = pathlib.Path().absolute() / file_path / f"train{suffix}.nc"
    with (xr.open_dataset(path) as train):
        #subset = train.sel(shot_id=shots[])
        data = train.load()

    print("\nDataset loaded successfully.")
    print("===============")
    print(data)
    print("===============")
    print(data.data_vars)
    print("===============")
    print(data["coil_current"].attrs)
    print("===============")

    