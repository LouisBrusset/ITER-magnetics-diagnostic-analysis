from pathlib import Path

from magnetics_diagnostic_analysis.ml_tools.pytorch_device_selection import select_torch_device

class Config:
    """Global variables configuration"""

    ### Useful variables
    valid_test_rates = [0.2, 0.25]

    ##########################################    
    ### Paths
    SUFFIX = "mscred"
    DIR_DATA = Path(__file__).absolute().parent.parent.parent.parent / "data"
    DIR_RAW_DATA = DIR_DATA / f"raw/{SUFFIX}"
    DIR_PREPROCESSED_DATA = DIR_DATA / f"preprocessed/{SUFFIX}"
    DIR_PROCESSED_DATA = DIR_DATA / f"processed/{SUFFIX}"
    DIR_MODEL_PARAMS = Path(__file__).absolute().parent.parent.parent.parent / f"results/model_params/{SUFFIX}"

    ### PyTorch device
    DEVICE = select_torch_device()

    ### Important parameters
    DATA_SHAPE = (3, 32, 32)
    WINDOW_SIZES = [10, 20, 50]
    GAP_TIME = 10  # Step to calculate the next window

    DATA_NUMBER = 20000
    SET_SEPARATIONS = [int(DATA_NUMBER * (1-valid_test_rates[1]) * (1-valid_test_rates[0])), int(DATA_NUMBER * (1-valid_test_rates[1]))]  # Train & Valid and Valid & Test split indices

    DEEP_CHANNEL_SIZES = [16, 32, 64]
    LSTM_NUM_LAYERS = 1
    LSTM_TIMESTEPS = 5
    LSTM_EFFECTIVE_TIMESTEPS = [1, 3, 4]

    ### Hyperparameters
    BATCH_SIZE = 10
    FIRST_LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5

    ### Train parameters
    N_EPOCHS = 50

    ### Data scrapping from MAST API
    RANDOM_SEED = 42
    GROUPS = ["magnetics"]
    STEADY_STATE = False  # Whether to apply steady state filtering or not
    RAW_DATA_FILE_NAME = f"data_{'_'.join(GROUPS)}_{SUFFIX}.nc"

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