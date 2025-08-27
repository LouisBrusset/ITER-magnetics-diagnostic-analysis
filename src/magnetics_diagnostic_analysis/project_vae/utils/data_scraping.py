import sys
import json
import random as rd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data_downloading.data_downloading import load_data, shot_list

if __name__ == "__main__":

    n_samples = 250
    random_seed = 42
    campaign_number = ""

    #shots = shot_list(campaign=campaign_number, quality=True)

    path = Path(__file__).parent.parent / "results" / "result_lists_magnetics.json"
    with open(path, "r") as f:
        data = json.load(f)
        shots = data["good_shot_ids"]

    #rd.seed(random_seed)
    #rd.shuffle(shots)
    #shots = shots[:n_samples]
    print("Chosen shots: \n", shots)
    print("Type of shots: ", type(shots))

    
    load_data(
        shots=shots,
        groups=["magnetics"],
        permanent_state=False,
        train_test_rate=0.15,
        random_seed=random_seed,
        file_path="src/magnetics_diagnostic_analysis/data/",
        suffix="_vae",
        verbose=False
    )

    print("Data loading completed.")


    load_data(
        shots=shots,
        groups=["magnetics"],
        permanent_state=True,
        train_test_rate=0.15,
        random_seed=random_seed,
        file_path="src/magnetics_diagnostic_analysis/data/",
        suffix="_mscred",
        verbose=False
    )

    print("Data loading completed.")
