import numpy as np
import matplotlib.pyplot as plt
import torch

from pathlib import Path

from magnetics_diagnostic_analysis.project_mscred.setting_mscred import config
from magnetics_diagnostic_analysis.project_mscred.utils.dataloader_building import create_data_loaders
from magnetics_diagnostic_analysis.project_mscred.model.mscred import MSCRED
from magnetics_diagnostic_analysis.ml_tools.metrics import mscred_anomaly_score, mscred_loss_function


def load_model(model_name: str = config.BEST_MODEL_NAME):
    """
    Load a pre-trained model parameters.

    Args:
        model_name (str): The name of the model to load. Defaults to the best model name registered in config.

    Returns:
        mscred: The loaded model.
    """
    mscred = MSCRED(
        encoder_in_channel=config.DATA_SHAPE[0],
        deep_channel_sizes=config.DEEP_CHANNEL_SIZES,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        lstm_timesteps=config.LSTM_TIMESTEPS,
        lstm_effective_timesteps=config.LSTM_EFFECTIVE_TIMESTEPS
    )
    mscred.load_state_dict(torch.load(config.DIR_MODEL_PARAMS / f"{model_name}.pth"))
    return mscred



def find_anomaly_threshold(
    model, 
    data_loader: torch.utils.data.DataLoader, 
    beta: float = 1.5, 
    device: torch.device = config.DEVICE
) -> float:
    """
    Find the anomaly detection threshold based on the reconstruction errors on the validation set.
    We get the maximum of the reconstruction error on the validation set an then multiply by a safty factor (beta).
        The beta factor can be tuned based on the desired sensitivity of the anomaly detection.
        It represents how confident we are in the model's reconstruction ability on normal data.
        If beta is too high, we might miss anomalies (false negatives).
        If beta is too low, we might flag too many normal instances as anomalies (false positives).
        Put beta lower, when we trust that the validation set is very clean and representative of normal behavior.
        Put beta higher, when validation set might contain some anomalies or is not fully representative of normal behavior.
    With experience, we realize that we hat to truncate the timeserie to avoid some problems at the beginning of the timeserie.
        Probably due to the LSTM state initialization. Issue to investigate in the future.
        So we are going to ignore the first 2*config.LSTM_TIMESTEPS points in the timeserie.

    Args:
        model: The trained MSCRED model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        beta (float): Safety factor to scale the maximum reconstruction error. Defaults to 1.5.
        device (torch.device): The device to run the computations on. Defaults to the device specified in config.

    Returns:
        float: The calculated anomaly detection threshold.
    
    Nota bene:
        ano_scores is already a numpy array thanks to the mscred_anomaly_score function.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        anomalies_full = np.zeros(len(data_loader)*config.BATCH_SIZE)
        i = 0
        for x in data_loader:
            x = x.to(device)
            x_recon = model(x)
            ano_scores = mscred_anomaly_score(x_recon, x)

            anomalies_full[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE] = ano_scores
            i += 1

    # Truncate the beginning of the timeserie to avoid initialization issues
    anomalies_full_tr = anomalies_full[2*config.LSTM_TIMESTEPS:]
    max_error = np.max(anomalies_full_tr)
    threshold_value = max_error * beta
    return threshold_value, anomalies_full_tr


def detect_anomalies_all(
    model, 
    data_loader: torch.utils.data.DataLoader, 
    threshold: float = 1.0, 
    device: torch.device = config.DEVICE
) -> float:
    """
    Detect anomalies in a test timeserie based on the anomaly scores and a given threshold (computes on a validation set).
    We also have the same truncation issue at the beginning of the timeserie (test set instead of validation set).
        Issue explain in the function find_anomaly_threshold.

    Args:
        model: The trained MSCRED model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
        threshold (float): The threshold for detecting anomalies. Defaults to 1.0.
        device (torch.device): The device to run the computations on. Defaults to the device specified in config.

    Returns:
        float: The calculated anomaly detection threshold.

    Nota bene:
        ano_scores is already a numpy array thanks to the mscred_anomaly_score function.
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        anomalies_full = np.zeros(len(data_loader)*config.BATCH_SIZE)
        reconstructed_full = np.zeros((len(data_loader)*config.BATCH_SIZE, *config.DATA_SHAPE))
        i = 0
        for x in data_loader:
            x = x.to(device)
            x_recon = model(x)
            reconstructed_full[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE] = x_recon.cpu().numpy()

            ano_scores = mscred_anomaly_score(x_recon, x)
            anomalies_full[i*config.BATCH_SIZE : (i+1)*config.BATCH_SIZE] = ano_scores
            i += 1

    ano_mask = anomalies_full > threshold
    return reconstructed_full, anomalies_full, ano_mask


def plot_anomalies_all(
    anomalies: np.ndarray,
    valid_anomaly_scores: np.ndarray,
    ano_mask: np.ndarray, 
    threshold: float, 
    start: int = 0, 
    end: int =  None
) -> None:
    """
    Plot the anomaly scores over time with the detected anomalies highlighted.

    Args:
        anomalies (np.ndarray): Array of anomaly scores.
        ano_mask (np.ndarray): Boolean array indicating detected anomalies.
        threshold (float): The threshold used for detecting anomalies.
        start (int): The starting index for the plot. Defaults to 0.
        end (int): The ending index for the plot. Defaults to None (i.e., the end of the array).

    Returns:
        None
    """
    plt.figure(figsize=(10, 5))
    plt.plot(anomalies[start:end], label='Anomaly Score', color='blue')
    #plt.plot(valid_anomaly_scores[start:end], label='Validation Score', color='green')
    plt.axhline(y=threshold, color='k', linestyle='--', label='Threshold (%.2f)' % threshold)
    legend_added = False
    for i in range(len(anomalies)):
        if i>=start and (end is None or i<end):
            if ano_mask[i]:
                if not legend_added:
                    plt.axvspan(i-0.5, i+0.5, color='red', linewidth=2, alpha=0.2, label='Anomalies')
                    legend_added = True
                else:
                    plt.axvspan(i-0.5, i+0.5, color='red', linewidth=2, alpha=0.2)
    plt.title('Anomaly Scores Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = Path(__file__).absolute().parent.parent.parent.parent.parent / "results/figures/mscred/anomaly_scores.png"
    plt.savefig(path)
    return None

def compute_residuals(
    reconstructed: np.ndarray,
    original: np.ndarray
) -> np.ndarray:
    """
    Compute the residuals between the reconstructed data and the original data.

    Args:
        reconstructed (np.ndarray): The reconstructed data from the model.
        original (np.ndarray): The original input data.

    Returns:
        np.ndarray: The computed residuals.
    """
    residuals = np.abs(original - reconstructed)
    return residuals


def detect_problematic_diagnostics(
    residuals: np.ndarray,
    anomaly_mask: np.ndarray,
    threshold_percentile: float = 90.
) -> tuple[list[int], np.ndarray]:
    """
    Detect problematic diagnostics based on the reconstruction residuals.

    Args:
        residuals (np.ndarray): The reconstruction residuals.
        anomaly_mask (np.ndarray): Boolean array indicating detected anomalies.
        threshold_percentile (float): The threshold for considering a diagnostic as problematic. Defaults to 90.

    Returns:
        list: List of indices of problematic diagnostics.
    """
    # Selection of anomalous time points and first window
    anomalous_residuals = residuals[anomaly_mask, 0, :, :]
    
    if anomalous_residuals.size == 0:
        return [], np.array([])
    
    total_residuals_matrix = np.sum(anomalous_residuals, axis=0)
    
    num_sensors = residuals.shape[2]
    sensor_scores = np.zeros(num_sensors)
    
    for sensor_i in range(num_sensors):
        # Get rows and columns involving sensor i
        residuals_involving_i = np.concatenate([
            total_residuals_matrix[sensor_i, :],
            total_residuals_matrix[:, sensor_i]
        ])
        sensor_scores[sensor_i] = np.mean(residuals_involving_i)
    
    # Threshold on scores to identify problematic sensors
    threshold = np.percentile(sensor_scores, threshold_percentile)
    problematic_sensors = np.where(sensor_scores > threshold)[0].tolist()
    
    return problematic_sensors, sensor_scores







if __name__ == "__main__":
    torch.backends.cudnn.enabled = False        # problem to fix in the future

    device = config.DEVICE
    mscred = load_model(model_name=config.BEST_MODEL_NAME)
    print(f"\nModel loaded on device: {device}")

    #data = np.random.rand(2000, *config.DATA_SHAPE)
    path = config.DIR_PREPROCESSED_DATA / "signature_matrices.npy"
    data = np.load(path)
    _, valid_loader, test_loader = create_data_loaders(
        data=data,
        batch_size=config.BATCH_SIZE,
        set_separations=config.SET_SEPARATIONS,
        gap_time=config.GAP_TIME,
        device=device
    )
    print("\nData loaders created.")

    ano_threshold, valid_anomaly_scores = find_anomaly_threshold(
        mscred, 
        valid_loader,
        beta=0.2,               # retrain MSCRED with bigger anomaly too set beta between 1 and 2
        device=device
    )
    print(f"\nAnomaly detection threshold: {ano_threshold:.4f}")

    reconstructed, anomalies, ano_mask = detect_anomalies_all(
        mscred, 
        test_loader, 
        threshold=ano_threshold, 
        device=device
    )
    print(f"\nDetected {np.sum(ano_mask)} anomalies in the test set.")

    plot_anomalies_all(
        anomalies, 
        valid_anomaly_scores,
        ano_mask, 
        threshold=ano_threshold, 
        start=0, 
        end=None
    )
    print("\nAnomaly scores plot saved.")


    residuals = compute_residuals(reconstructed, data[config.SET_SEPARATIONS[1]//config.BATCH_SIZE:])
    print("\nResiduals computed.")

    problematic_sensors, sensor_scores = detect_problematic_diagnostics(
        residuals,
        ano_mask,
        threshold_percentile=90
    )
    print(f"\nDetected problematic sensors: {problematic_sensors}")
    print(f"Sensor scores: {sensor_scores}")











