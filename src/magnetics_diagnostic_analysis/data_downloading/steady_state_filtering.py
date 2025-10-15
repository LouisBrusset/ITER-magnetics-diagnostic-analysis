import numpy as np
import xarray as xr
from scipy import signal


def _keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected True component in a boolean mask.

    Args:
        mask (np.ndarray): Input boolean array

    Returns:
        np.ndarray: Boolean array with only the largest connected True component
    """
    # Find the start and end indices of True regions
    diff = np.diff(mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    # Case when the mask starts or ends with True
    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))

    # If no True regions found, return the original mask
    if len(starts) == 0:
        return mask

    # Find biggest connected component
    lengths = ends - starts
    largest_idx = np.argmax(lengths)
    new_mask = np.zeros_like(mask)
    new_mask[starts[largest_idx] : ends[largest_idx]] = True

    return new_mask


def _filter_low_current_regions(
    ip: np.ndarray, mask: np.ndarray, min_current: float = 4.0e4
) -> np.ndarray:
    """
    Remove True values from mask where the current is below a threshold.

    Args:
        ip (np.ndarray): 1D array of current values
        mask (np.ndarray): Boolean mask to filter
        min_current (float): Minimum current threshold

    Returns:
        np.ndarray: Filtered boolean mask
    """
    new_mask = mask.copy()
    new_mask[np.abs(ip) < min_current] = False
    return new_mask


def ip_filter(
    ip: np.ndarray, filter: str = "default", min_current: float = None
) -> np.ndarray:
    """
    Create a mask to select the permanent state phase of each shot.

    Args:
        ip (np.ndarray): 1D array of current values for one shot.
        filter (str): Type of filter to apply. 'default' uses a low-pass FIR filter. Else 'bidirectional'.
        min_current (float): Minimum current threshold for filtering

    Returns:
        tuple: A tuple containing:
            - mask_clean (np.ndarray): A boolean mask where True indicates the permanent state phase.
            - flat_mask (np.ndarray): initial flat regions mask before post-processing
            - filtered_ip (np.ndarray): filtered current signal
    """
    # 1. Low-pass filtering FIR (=Finite Impulse Response)
    cutoff = 1.0e-3  # cutoff frequency (Hz)
    ntaps = 101  # Number of coefficients (must be odd for symmetry)
    fir_coeff = signal.firwin(ntaps, cutoff, window="hamming")

    if filter == "bidirectional":
        filtered_ip = signal.filtfilt(fir_coeff, 1.0, ip)
    else:
        filtered_ip = signal.lfilter(fir_coeff, 1.0, ip)
        delay = (ntaps - 1) // 2
        filtered_ip = filtered_ip[delay:]
        filtered_ip = np.pad(
            filtered_ip, (0, delay), mode="edge"
        )  # RRepeat last value to maintain original length

    # 2. Gradient calculation
    gradient = np.gradient(filtered_ip)

    # 3. Thresholding for flat regions
    std_gradient = np.std(gradient)
    threshold = (
        0.25 * std_gradient
    )  # Adaptative threshold based on gradient standard deviation
    flat_mask = np.abs(gradient) < threshold

    # 4. Optional current thresholding
    if min_current is not None:
        flat_mask = _filter_low_current_regions(filtered_ip, flat_mask, min_current)

    # 5. Keep largest connected component
    mask_clean = _keep_largest_connected_component(flat_mask)

    return mask_clean, filtered_ip, fir_coeff


if __name__ == "__main__":
    index = [
        28752,
        28750,
        28655,
        28656,
        28657,
        28744,
        28751,
        28747,
        28748,
        28749,
        28755,
        28757,
        28758,
        28763,
        28801,
        28764,
        28765,
        28766,
        28767,
        28768,
        28769,
        28770,
        28771,
        28772,
        28773,
    ]
    shot_index = np.random.choice(index)
    ip_example = xr.open_zarr(
        f"https://s3.echo.stfc.ac.uk/mast/level2/shots/{shot_index}.zarr",
        group="summary",
    )["ip"]

    ip_values = ip_example.values
    time_values = (
        ip_example.coords["time"].values
        if "time" in ip_example.coords
        else np.arange(len(ip_values))
    )

    mask, filtered_signal, fir_coeff = ip_filter(
        ip_values, filter="default", min_current=4.0e4
    )

    # Display the result
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(time_values, ip_values, label="Current (IP)")
    plt.plot(time_values, filtered_signal, label="Filtered Signal", linestyle="-.")
    plt.plot(
        time_values,
        mask * np.max(ip_values),
        label="Permanent State Mask",
        linestyle="--",
    )
    plt.legend()
    plt.title("Permanent State Filtering Example")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (A)")

    # Display filter
    plt.figure(figsize=(12, 6))
    w, h = signal.freqz(fir_coeff)
    plt.plot(w, 20 * np.log10(np.abs(h)))
    plt.title("Filter Frequency Response")
    plt.xlabel("Normalized Frequency (×π rad/sample)")
    plt.ylabel("Gain (dB)")

    plt.show()
