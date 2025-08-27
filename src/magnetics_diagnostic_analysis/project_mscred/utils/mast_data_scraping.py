import json
import random as rd
from pathlib import Path

from magnetics_diagnostic_analysis.project_mscred.setting_mscred import config
from magnetics_diagnostic_analysis.data_downloading.data_downloading import load_data, select_good_vars
from magnetics_diagnostic_analysis.data_downloading.data_washing import clean_data

def data_scraping_mscred():
    # Configuration & Shots choice
    # shots = shot_list(campaign=campaign_number, quality=True)
    path = Path(__file__).absolute().parent.parent.parent.parent.parent / "notebooks/result_files/nan_stats_magnetics/result_lists_magnetics_nans.json"
    with open(path, "r") as f:
        data = json.load(f)
    shots = data["good_shot_ids"]
    print("Chosen shots: \n", shots)
    print("Type of shots: ", type(shots))
    print("Number of shot: ", len(shots))

    vars = data["good_var_ids"]
    print("Chosen vars: \n", vars)
    print("Type of vars: ", type(vars))
    print("Number of var: ", len(vars))

    

    load_data(
        shots=shots,
        groups=config.GROUPS,
        steady_state=config.STEADY_STATE,
        verbose=False
    )
    print("Data loading for MSCRED completed.")


    clean_data(
        group="magnetics",
        suffix=config.SUFFIX,
        verbose=True
    )
    print("Dataset washing for MSCRED completed.")



if __name__ == "__main__":
    data_scraping_mscred()