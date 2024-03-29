import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pathlib

from functools import reduce

class Rixs:
    def __init__(
        self,
        spec_dir,
        info_file,
        **kwargs
    ):
        self.spec_dir = pathlib.Path(spec_dir, **kwargs)
        data_list = []
        self.child_list = self.spec_dir.glob('*-1D.txt')
        self.child_list = sorted(self.child_list)
        for c in self.child_list:
                data_list.append(pd.read_csv(c, skiprows=9, sep=r'\t', engine='python'))
            
        self.data_list = data_list
        self.df = pd.concat([d.set_index('X') for d in self.data_list], axis=1, join='inner').reset_index()
        
        self.info_df = pd.read_csv(pathlib.Path(info_file, **kwargs), skiprows=12, sep=r'\t', engine='python')