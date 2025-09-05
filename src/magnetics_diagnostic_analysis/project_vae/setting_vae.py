from pathlib import Path
# import yaml

from magnetics_diagnostic_analysis.ml_tools.pytorch_device_selection import select_torch_device

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
    DIR_MODEL_PARAMS = Path(__file__).absolute().parent.parent.parent.parent / f"results/model_params/{SUFFIX}"
    DIR_FIGURES = Path(__file__).absolute().parent.parent.parent.parent / f"results/figures/{SUFFIX}"

    ### PyTorch device
    DEVICE = select_torch_device(temporal_dim="sequential")
    SEED = 42

    ### Important parameters
    MULTIVARIATE = False    # Whether to use multivariate data or not
    MAX_LENGTH = 3000       # Max length of the sequences (in time steps)
    N_SUBSAMPLE = 20        # Factor to subsample the data
    N_CHAN_TO_KEEP = None   # Number of channels to keep if multivariate
    VAR_NAME = "ip"         # Variable name to use if univariate

    DATA_NUMBER = 12877819  # Total number of data points to consider
    SET_SEPARATION =  int(DATA_NUMBER * (1-train_test_rates))  # Train & Test split indice

    ### VAE architecture
    LATENT_DIM = 128
    LSTM_HIDDEN_DIM = 256
    LSTM_NUM_LAYERS = 2
    BETA = 2.0

    ### Hyperparameters
    BATCH_SIZE = 1000
    FIRST_LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5     # if needed

    ### Train parameters
    N_ITERATIONS = 2
    N_EPOCHS = 5
    KDE_PERCENTILE_RATE = 0.05
    DBSCAN_EPS = 0.0001
    DBSCAN_MIN_SAMPLES = 30
    KNN_N_NEIGHBORS = 10

    ### Data scrapping from MAST API
    RANDOM_SEED = 42
    GROUPS = ["magnetics"]
    STEADY_STATE = False  # Whether to apply steady state filtering or not
    RAW_DATA_FILE_NAME = f"data_{'_'.join(GROUPS)}.nc"

    ### Others
    BEST_MODEL_NAME = "model1"

    # Method to update parameters
    @classmethod
    def update(cls, **kwargs):
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)

# Global instance
config = Config()



# update data number
#config.update(DATA_NUMBER=10000000)