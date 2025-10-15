from pathlib import Path

# import yaml

from magnetics_diagnostic_analysis.ml_tools.random_seed import seed_everything
from magnetics_diagnostic_analysis.ml_tools.pytorch_device_selection import (
    select_torch_device,
)


class Config:
    """Global variables configuration"""

    ### Useful variables
    train_test_rates = 0.25

    ##########################################
    ### Paths
    SUFFIX = "vae"
    DIR_DATA = Path(__file__).absolute().parent.parent.parent.parent / "data"
    DIR_RAW_DATA = DIR_DATA / f"raw"
    DIR_PREPROCESSED_DATA = DIR_DATA / f"preprocessed/{SUFFIX}"
    DIR_PROCESSED_DATA = DIR_DATA / f"processed/{SUFFIX}"
    DIR_MODEL_PARAMS = (
        Path(__file__).absolute().parent.parent.parent.parent
        / f"results/model_params/{SUFFIX}"
    )
    DIR_FIGURES = (
        Path(__file__).absolute().parent.parent.parent.parent
        / f"results/figures/{SUFFIX}"
    )
    for direction in [
        DIR_DATA,
        DIR_RAW_DATA,
        DIR_PREPROCESSED_DATA,
        DIR_PROCESSED_DATA,
        DIR_MODEL_PARAMS,
        DIR_FIGURES,
    ]:
        direction.mkdir(parents=True, exist_ok=True)

    ### PyTorch device
    DEVICE = select_torch_device(temporal_dim="sequential")
    SEED = 42

    ### Important parameters
    MULTIVARIATE = False  # Whether to use multivariate data or not
    MAX_LENGTH = 3000  # Max length of the sequences (in time steps)
    N_SUBSAMPLE = 1  # Factor to subsample the data
    N_CHAN_TO_KEEP = None  # Number of channels to keep if multivariate
    VAR_NAME = "ip"  # Variable name to use if univariate

    DATA_NUMBER = 12877819  # Total number of data points to consider
    SET_SEPARATION = int(
        DATA_NUMBER * (1 - train_test_rates)
    )  # Train & Test split indice

    ### VAE architecture
    LATENT_DIM = 64
    LSTM_HIDDEN_DIM = 200
    LSTM_NUM_LAYERS = 2
    LSTM_BPTT_STEPS = None
    BETA = 1.5

    ### Hyperparameters
    BATCH_SIZE = 32
    FIRST_LEARNING_RATE = 0.0003
    WEIGHT_DECAY = 1e-5  # if needed

    ### Train parameters
    N_ITERATIONS = 1
    N_EPOCHS = 13
    KDE_PERCENTILE_RATE = 0.05
    DBSCAN_EPS = 0.0005
    DBSCAN_MIN_SAMPLES = 15
    KNN_N_NEIGHBORS = 10

    ### Data scrapping from MAST API
    GROUPS = ["magnetics"]
    STEADY_STATE = False  # Whether to apply steady state filtering or not
    RAW_DATA_FILE_NAME = f"data_{'_'.join(GROUPS)}.nc"

    ### Others
    BEST_MODEL_NAME = "model1"

    ### Set seed for reproducibility
    seed_everything(SEED)

    # Method to update parameters
    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)


# Global instance
config = Config()


# update data number
# config.update(DATA_NUMBER=10000000)
