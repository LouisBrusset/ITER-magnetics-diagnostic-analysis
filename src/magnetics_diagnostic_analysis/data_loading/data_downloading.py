import numpy as np
import pandas as pd
import xarray as xr

import pathlib
import tqdm
import requests
import pickle

from permanent_state_filtering import ip_filter


def shot_list(campaign: str = "M9", quality: bool = None) -> list[int]:
    """
    Return a list of shot IDs for a given campaign.

    Parameters
    campaign: Campaign name to filter shots.
    quality: If True, return only shots with 'cpf_useful' label. If False, return shots with 'cpf_abort' label.
    
    Returns
    List of shot IDs.
    """
    URL = "https://mastapp.site"
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
        group=group,
    )



def build_level_2_data(shots: list[int], groups: list[str], permanent_state: bool = False) -> xr.Dataset:
    """
    Retrieve specified groups of diagnostics from shots in the M9 campaign during permanent state or not.
    
    Parameters
    groups: List of diagnostic groups to retrieve. If None, all groups are retrieved.
    permanent_state: If True, only retrieve shots during the permanent state phase of the campaign.

    Return
    An xarray Dataset containing the requested diagnostic data.
    """
    # URL = "https://mastapp.site"
    # summary = pd.read_parquet(f'{URL}/parquet/level2/shots')
    # summary = summary.loc[:, ["shot_id", 'cpf_useful', 'cpf_abort']]

    dataset = []

    for shot in tqdm.tqdm(shots, desc="Loading shots", total=len(shots)):
        summary_ip = to_dask(shot, "summary")['ip']
        time_ip = summary_ip['time']
        if permanent_state:
            mask, _, _ = ip_filter(summary_ip.values, filter='default', min_current=4.e4)
            time_ref = time_ip[mask]
        else:
            time_ref = time_ip
        
        signals = []

        for group in groups:
            try:
                data = to_dask(shot, group)
                if group == "summary":
                    # If the group is summary, we do not need the 'ip' variable
                    data = data.drop_vars("ip", errors="ignore")
            
                data = data.interp({"time": time_ref})

                other_times = set()
                for var in data.data_vars:
                    time_dim = next((dim for dim in data[var].dims if dim.startswith('time')), 'time')
                    if time_dim != "time":
                        other_times.add(time_dim)
                    data[var] = data[var].interp({time_dim: time_ip})               
                    data[var] = data[var].transpose("time", ...)
                    data[var].attrs |= {"group": group}
                data = data.drop_vars(other_times)

                # if permanent_state:
                #     data = data.where(mask, drop=True)
                # data = data.expand_dims("group")
                # data = data.assign_coords(group=("group", [group]))
                
                signals.append(data)

            except (IndexError, KeyError):
                print(f"Group {group} not found for shot {shot}. Skipping.")

        if not signals:
            continue

        shot_data = xr.merge(signals, combine_attrs="drop_conflicts")
        shot_data = shot_data.expand_dims("shot_id")
        shot_data = shot_data.assign_coords(shot_id=("shot_id", [shot]))

        dataset.append(shot_data)

    final = xr.concat(dataset, dim="shot_id", combine_attrs="drop_conflicts", coords="minimal")
    return final






def load_data(shots: list[int], groups: list[str], permanent_state: bool) -> xr.Dataset:
    """
    Load data from cache or build it if not available.

    Parameters
    shots: List of shot IDs to retrieve data for.
    groups: List of diagnostic groups to retrieve data from.
    permanent_state: If True, only retrieve shots during the permanent state phase of the campaign.

    Return
    An xarray Dataset containing the requested diagnostic data.
    """
    path = pathlib.Path().absolute() / "src/magnetics_diagnostic_analysis/data"
    filename = path / "brut_data.pkl"
    pathlib.Path(filename).parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(filename, "rb") as f:
            brut_data = pickle.load(f)
    except FileNotFoundError:
        brut_data = build_level_2_data(
            shots=shots, 
            groups=groups, 
            permanent_state=permanent_state
            )
        with open(filename, "wb") as f:
            pickle.dump(brut_data, f)
    return brut_data



if __name__ == "__main__":
    shots = shot_list(campaign="M9", quality=True)[:400]
    groups = ["summary", "magnetics", "spectrometer_visible", "pf_active"]
    permanent_state = True

    brut_data = load_data(shots=shots, groups=groups, permanent_state=permanent_state)
    
    path = pathlib.Path().absolute() / "src/magnetics_diagnostic_analysis/data"
    brut_data.to_netcdf(path / "train.nc")

    with (xr.open_dataset(path / "train.nc") as train):
        subset = train.isel(shot_id=shots[2:5])
        data = subset.load()

    print("===============")
    print(data.coords)
    print("===============")
    print(data.data_vars)
    print("===============")
    print(data.attrs)
    print("===============")
    print(data['b_field_pol_probe_ccbv_field'].sel(shot_id=shots[2]).values)
    print("===============")
    print(data['density_gradient'].sel(shot_id=shots[2]))
    print("===============")
    





#test_size = 0.3  # fraction of dataset to use for testing
#dataset = load_data()[(448, 640)]
#dataset = dataset.drop_vars(["time", "shot_id"])  # anonymize dataset 
## shuffle dataset
#shot_index = np.arange(dataset.sizes["shot_id"], dtype=int)
#rng = np.random.default_rng(7)
#rng.shuffle(shot_index)
#test_split = int(np.floor(test_size * dataset.sizes["shot_id"]))
#train = dataset.isel(shot_id=shot_index[test_split:])
#test = dataset.isel(shot_id=shot_index[:test_split])
#solution = test.volume.to_pandas().to_frame()
#rng.random(len(solution))
#solution["Usage"] = np.where(rng.random(len(solution)) < 0.5, "Public", "Private")
#test = test.drop_vars("volume")  # drop target from test dataset
## write datasets to file
#path = pathlib.Path().absolute().parent / "fair_mast_data/plasma_volume"
#train.to_netcdf(path / "train.nc")
#test.to_netcdf(path / "test.nc")
#solution.to_csv(path / "solution.csv")

############################################### 


#def to_dataset(shots: pd.Series):
#    """Return a concatenated xarray Dataset for a series of input shots."""
#    dataset = []
#    for shot_index, shot_id in shots.items():
#        target = to_dask(shot_id, "equilibrium")['psi']
#        signal = []
#        for group in ["magnetics", "spectrometer_visible", "soft_x_rays", "thomson_scattering"]: 
#            data = to_dask(shot_id, group).interp({"time": target.time})
#            if "major_radius" in data:
#                data = data.interp({"major_radius": target.major_radius})
#            other_times = set()
#            for var in data.data_vars:  # Interpolate to the target time
#                time_dim = next((dim for dim in data[var].dims 
#                                 if dim.startswith('time')), 'time')
#                if time_dim != "time":
#                    other_times.add(time_dim)
#                data[var] = data[var].interp({time_dim: target.time})               
#                data[var] = data[var].transpose("time", ...)
#                data[var].attrs |= {"group": group}
#            data = data.drop_vars(other_times)
#            signal.append(data)
#        signal = xr.merge(signal, combine_attrs="drop_conflicts")
#        signal["shot_index"] = "time", shot_index * np.ones(target.sizes["time"], int)
#        dataset.append(xr.merge([signal, target], combine_attrs="override"))
#    return xr.concat(dataset, "time", join="override", combine_attrs="drop_conflicts")
#
#source_ids = np.array([15585, 15212, 15010, 14998, 30410, 30418, 30420])
#
#rng = np.random.default_rng(7)
#rng.shuffle(source_ids)
#source_ids = pd.Series(source_ids)
#
#split_ids = {
#    "train": source_ids[:5],
#    "test": source_ids[5:],
#}
#
#dataset = {mode: to_dataset(shot_ids) for mode, shot_ids in split_ids.items()}
#
## extract solution
#psi = dataset["test"].psi.data.reshape((dataset["test"].sizes["time"], -1))
#solution = pd.DataFrame(psi)
#solution.index.name = "index"
#shot_index = dataset["test"].shot_index.data
#solution["Usage"] = [{5: "Public", 6: "Private"}.get(index) for index in shot_index]
## delete solution from test file
#dataset["test"] = dataset["test"].drop_vars("psi")
#
## write to file
#path = pathlib.Path().absolute().parent / "fair_mast_data/plasma_equilibrium"
#path.mkdir(exist_ok=True)
#dataset["train"].to_netcdf(path / "train.nc")
#dataset["test"].to_netcdf(path / "test.nc")
#solution.to_csv(path / "solution.csv")


