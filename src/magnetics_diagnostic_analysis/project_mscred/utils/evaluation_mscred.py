import numpy as np
import matplotlib.pyplot as plt
import torch

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






def find_anomaly_threshold(model, data_loader: torch.utils.data.DataLoader, beta: float = 1.5, device: torch.device = config.DEVICE):
    """
    Find the anomaly detection threshold based on the reconstruction errors on the validation set.
    We get the maximum of the reconstruction error on the validation set an then multiply by a safty factor (beta).
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

    max_error = np.max(anomalies_full)
    threshold_value = max_error * beta
    return threshold_value







if __name__ == "__main__":

    device = config.DEVICE
    mscred = load_model(model_name=config.BEST_MODEL_NAME)

    data = np.random.rand(2000, *config.DATA_SHAPE)

    _, valid_loader, test_loader = create_data_loaders(data)
























