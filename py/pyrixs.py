import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pathlib
import warnings

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
        plot_pfy=True,
        pfy_color='gray',
        plot_tfy=True,
        tfy_color='black',
        plot_tey=True,
        tey_color='blue',
        izero='Izero',
        drop=None
    ):
        fontsize=12
        self.fig, self.axs = plt.subplots(1, 2, layout='constrained', gridspec_kw={'width_ratios': [1, 0.25]})

        if drop is None:
            drop = []

        if self.info_df[izero].min() == 0:
            idx = self.info_df[izero].argmin()
            if idx not in drop:
                drop.append(idx)
            warnings.warn('I0 = 0 detected at row {}'.format(idx))
            
        if len(drop) > 0:
            column_numbers = [x for x in range(self.df.shape[1])]  # list of columns' integer indices
            for i in drop:
                column_numbers.remove(i) #removing column integer index idx
            self.df = self.df.iloc[:, column_numbers] #return all columns except the idx-th column
            self.info_df.drop(drop, inplace=True, axis=0)
            warnings.warn('Deleted row(s) {}.'.format(drop))
            
        I0 = np.array(self.info_df[izero])
        x = np.array(self.df['X'])
        y = np.array(self.info_df['BL 8 Energy'])
        Z = np.array(self.df.iloc[:, 1:]).transpose()
        
        # print(I0)
        
        for i in range(len(I0)):
            Z[:,i] = Z[:,i]/I0[i]
            
        # print(Z.argmax())
        Z = (Z-Z.min())/(Z.max()-Z.min())
        # print(Z.max(keepdims=True))
        
        pc = self.axs[0].pcolormesh(x, y, Z, linewidth=0, antialiased=True, alpha=1, edgecolor='face', rasterized=True)

        # print(self.df.columns)

        # temp_df = self.df.drop(labels='X', axis=1)

        I = Z.sum(axis=1)

        tfy = np.array(self.info_df['TFY'])
        tey = np.array(self.info_df['TEY'])

        if plot_pfy:
            self.axs[1].plot((I-min(I))/(max(I)-min(I)), y, color=pfy_color)
        if plot_tfy:
            self.axs[1].plot((tfy-min(tfy))/(max(tfy)-min(tfy)), y, color=tfy_color)
        if plot_tey:
            self.axs[1].plot((tey-min(tey))/(max(tey)-min(tey)), y, color=tey_color)

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
        if show:
            plt.show()
        if savefig:
            plt.savefig(savefig)