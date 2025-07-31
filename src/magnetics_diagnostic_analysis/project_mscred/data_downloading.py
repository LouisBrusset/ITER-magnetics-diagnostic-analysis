import sys
import random as rd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_loading.data_downloading import load_data, shot_list

if __name__ == "__main__":

    n_samples = 300
    random_seed = 42
    campaign_number = ""

    shots = shot_list(campaign=campaign_number, quality=True)
    rd.seed(random_seed)
    rd.shuffle(shots)
    shots = shots[:n_samples]
    print("Chosen shots: \n", shots)
    print("Type of shots: ", type(shots))

    
    load_data(
        shots=shots,
        groups=["summary", "magnetics"],
        permanent_state=False,
        train_test_rate=0.3333,
        random_seed=random_seed,
        file_path="src/magnetics_diagnostic_analysis/data/",
        suffix="_mscred"
    )

