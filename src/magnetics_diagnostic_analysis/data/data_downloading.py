import numpy as np
import pandas as pd
import xarray as xr

import pathlib
import tqdm
import requests



def defined_url(shot: int, group: str, level: int = 2) -> str:
    """Return the URL for a given shot, group, and level."""
    shot_data = requests.get(f"https://mastapp.site/json/shots/{shot}").json()
    endpoint, url = shot_data["endpoint_url"], shot_data["url"]
    shot_url = url.replace("s3:/", endpoint)
    return shot_url

def to_dask(shot: int, group: str, level: int = 2) -> xr.Dataset:
    """Return a Dataset from the MAST Zarr store."""
    url = defined_url(shot, group, level)
    return xr.open_zarr(
        f"https://s3.echo.stfc.ac.uk/mast/level{level}/shots/{shot}.zarr",
        group=group,
    )