import numpy as np
import pandas as pd
import pathlib
import warnings
import xarray as xr

import matplotlib.pyplot as plt

__author__ = 'Isaac Zakaria'

def main():
    expt_dir = '../data/irixs/2024-6-6/CCD Scan 7009'
    expt_path = pathlib.Path(expt_dir)

    # read and clean Andor info (AI) file
    ai_path = sorted(expt_path.glob('*AI.txt'))
    if len(ai_path) > 1:
        warnings.warn('More than one Andor info file found.')
    ai_path = ai_path[0]
    ai_df = pd.read_table(ai_path, skiprows=12)
    ds = xr.Dataset(ai_df)
    ds = ds.assign_coords({'excitation_energy': ds['BL 8 Energy']})
    ds = ds.swap_dims({'dim_0': 'excitation_energy'})
    ds = ds.drop_vars('dim_0')
    #print(ds)

    # read XES/RIXS traces
    xes_path = expt_path.walk()
    print(xes_path)

if __name__ == '__main__':
    main()
