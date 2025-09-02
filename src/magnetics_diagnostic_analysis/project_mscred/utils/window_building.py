import numpy as np
import xarray as xr

from pathlib import Path
import json

from magnetics_diagnostic_analysis.project_mscred.setting_mscred import config
from magnetics_diagnostic_analysis.project_mscred.utils.matrix_generator import generate_signature_matrix   
from magnetics_diagnostic_analysis.project_mscred.utils.synthetic_anomaly_adding import create_anomalies, plot_anomalies


def select_data_channels(ds, n_to_keep: int = 32, mandatory_channels: list[str] = []) -> np.ndarray:
    """Select specific data channels from the dataset based on the provided groups."""
    # Load possible channels
    path = Path(__file__).absolute().parent.parent.parent.parent.parent / "notebooks/result_files/nan_stats_magnetics/result_lists_magnetics_nans.json"
    with open(path) as f:
        d = json.load(f)
    possible_channels = d["good_vars_ids"]

    # Choose channels to keep
    channels_to_keep = []
    for ch in mandatory_channels:
        channels_to_keep.append(ch)
    while len(channels_to_keep) < n_to_keep:
        choosen = np.random.choice(possible_channels)
        if choosen not in channels_to_keep:
            channels_to_keep.append(choosen)

    data_return = np.zeros((n_to_keep, ds.time.size))

    for i, var_ch in enumerate(channels_to_keep):
        
        if "::" in var_ch:
            var, ch = var_ch.split("::")
            coord = [ c for c in ds[var].dims if c != "time"][0]
            print(f"Coordinate for variable {var} channel {ch}: {coord}")

            vals = ds[var].sel(**{coord: ch}).values
            print(f"Data shape for variable {var} channel {ch}: {vals.shape}")

        else:
            var = var_ch
            print(f"\nSelecting variable {var} (no channel)")

            vals = ds[var].values
            print(f"Data shape for variable {var}: {vals.shape}")

        data_return[i, :] = vals

    return data_return, channels_to_keep


def build_windows():
    # Load preprocessed data
    data_path = config.DIR_PREPROCESSED_DATA / f"data_magnetics_{config.SUFFIX}_cleaned.nc"
    ds = xr.open_dataset(data_path)
    ds = ds.isel(time=slice(0, config.DATA_NUMBER))  # Select a subset for faster processing during testing
    print(f"Data loaded from {data_path}")

    # Select specific channels
    data, channels_to_keep = select_data_channels(
        ds,
        n_to_keep=config.DATA_SHAPE[1],
        mandatory_channels=[
            "ip",
            "flux_loop_flux::AMB_FL/CC07",
            "flux_loop_flux::AMB_FL/CC09",
            "flux_loop_flux::AMB_FL/P3L/4",
            "flux_loop_flux::AMB_FL/P3U/1",
            "b_field_tor_probe_saddle_voltage::XMB_SAD/OUT/M01",
            "b_field_tor_probe_saddle_voltage::XMB_SAD/OUT/M03",
        ]
    )
    path = config.DIR_PREPROCESSED_DATA / f"selected_channels_{config.SUFFIX}.json"
    with open(path, 'w') as f:
        json.dump(channels_to_keep, f, indent=4)
    print(f"Data shape after channel selection: {data.shape}")

    # Add synthetic anomalies
    data_anomalies, anomalies_info = create_anomalies(
        data, 
        start_index=config.SET_SEPARATIONS[1], 
        duration_range=(50, 500), 
        n_anomalies=20, 
        anomaly_strength=6.5, 
        seed=config.SEED
    )
    #plot_anomalies(data, data_anomalies, anomalies_info, n_series_to_plot=3)
    path = config.DIR_PREPROCESSED_DATA / f"created_anomalies_{config.SUFFIX}.json"
    with open(path, 'w') as f:
        json.dump(anomalies_info, f, indent=4)
    print(f"\nAnomalies added at indices: {anomalies_info}")

    # Generate signature matrix
    _ = generate_signature_matrix(
        data_anomalies,
        win_size=config.WINDOW_SIZES,
        min_time=0,
        max_time=None,
        gap_time=config.GAP_TIME,
        normalize=True,
        saving=True
    )
    print("\nSignature matrix generation completed.")

    print("\nWindows building process completed.")

if __name__ == "__main__":
    build_windows()

