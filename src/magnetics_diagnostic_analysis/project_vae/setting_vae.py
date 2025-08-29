from pathlib import Path
import yaml

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

    ### PyTorch device
    DEVICE = select_torch_device(temporal_dim="parallel")
    SEED = 42

    ### Important parameters
    MAX_SEQ_LENGTH = 100
    N_DIAGS = 96
    DATA_SHAPE = (MAX_SEQ_LENGTH, N_DIAGS)

    DATA_NUMBER = 10000000
    SET_SEPARATION =  int(DATA_NUMBER * (1-train_test_rates))  # Train & Test split indice

    LSTM_NUM_LAYERS = 2

    ### Hyperparameters
    BATCH_SIZE = 10
    FIRST_LEARNING_RATE = 0.005
    WEIGHT_DECAY = 1e-5     # if needed

    ### Train parameters
    N_EPOCHS = 200

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