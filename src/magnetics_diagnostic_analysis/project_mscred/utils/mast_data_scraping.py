import json
import random as rd
from pathlib import Path

from magnetics_diagnostic_analysis.project_mscred.setting_mscred import config
from magnetics_diagnostic_analysis.data_downloading.data_downloading import load_data

def data_scraping_mscred():
    # Configuration & Shots choice
    # shots = shot_list(campaign=campaign_number, quality=True)
    path = Path(__file__).absolute().parent.parent.parent.parent.parent / "notebooks/result_files/all_shots_magnetics/choosen_shots_to_load.json"
    with open(path, "r") as f:
        data = json.load(f)
        shots = data["good_shot_ids"]
    shots = shots[:20]  # For testing purpose, we take only 20 shots
    print("Chosen shots: \n", shots)
    print("Type of shots: ", type(shots))

    
    load_data(
        shots=shots,
        groups=config.GROUPS,
        steady_state=config.STEADY_STATE,
        verbose=False
    )

    print("Data loading for MSCRED completed.")



if __name__ == "__main__":
    data_scraping_mscred()