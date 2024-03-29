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
        
    def plot_mrixs(
        self,
        show=False,
        savefig=None,
    ):
        fontsize=12
        self.fig, self.axs = plt.subplots(1, 2, layout='constrained')
        
        x = self.df['X']
        y = self.info_df['BL 8 Energy']
        Z = np.array(self.df.iloc[:, 1:]).transpose()
        
        pc = self.axs[0].pcolormesh(x, y, Z, linewidth=0, antialiased=True, alpha=1, edgecolor='face', rasterized=True)

        # print(self.df.columns)

        temp_df = self.df.drop(labels='X', axis=1)

        I = temp_df.sum(axis=0)
        print(I)
        # print(self.df.sum(axis=1))

        tfy = self.info_df['TFY']
        tey = self.info_df['TEY']

        self.axs[1].plot((I-min(I))/(max(I)-min(I)), y)
        self.axs[1].plot((tfy-min(tfy))/(max(tfy)-min(tfy)), y)
        self.axs[1].plot((tey-min(tey))/(max(tey)-min(tey)), y)

        self.axs[0].set_xlim([0,2047])
        self.axs[1].set_xlim([-0.1,1.1])
        self.axs[0].set_ylim([775,788])
        self.axs[1].set_ylim([775,788])


        for a in self.axs:
            # a.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad*2/3)
            # a.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad*2/3)
            a.tick_params(labelsize=fontsize)
            # self.axs.xaxis.set_tick_params(width=tick_linewidth, length=tick_length, which='major')
            # self.axs.xaxis.set_tick_params(width=tick_linewidth, length=tick_length*0.5, which='minor')
            # self.axs.yaxis.set_tick_params(width=tick_linewidth, length=tick_length, which='major')
            # self.axs.yaxis.set_tick_params(width=tick_linewidth, length=tick_length*0.5, which='minor')
            
        self.axs[0].set_xlabel('Emission Energy (a.u.)', fontsize=fontsize)
        self.axs[1].set_xlabel('Norm. Intensity', fontsize=fontsize)

        self.axs[0].set_ylabel('Excitation Energy (eV)', fontsize=fontsize)
        # self.axs[1].ylabel('Norm. Intensity')
        if show:
            plt.show()
        if savefig:
            plt.savefig(savefig)