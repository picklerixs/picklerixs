import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pathlib
import warnings

from functools import reduce
from matplotlib.ticker import MultipleLocator

# TODO implement interactive mRIXS plots
# TODO make masking of XAS data user-specifiable and not jank
# TODO clean up plot styling options
class Rixs:        
    def __init__(
        self,
        spec_dir,
        info_file,
        **kwargs
    ):
        '''
        Methods for processing data from iRIXS.
        Generates a dataframe containing detector information (info_df) and a dataframe containing 1D spectral data (df).
        
        ARGS:
            spec_dir (str or pathlib.Path()): Directory containing 1D spectral data in .txt format.
            info_file (str or pathlib.Path()): Andor detector information file. 
            
        kwargs are passed to pd.read_csv() when attempting to open info_file.
        '''
        self.spec_dir = pathlib.Path(spec_dir, **kwargs)
        data_list = []
        self.child_list = self.spec_dir.glob('*-1D.txt')
        self.child_list = sorted(self.child_list)
        for c in self.child_list:
            data_list.append(pd.read_csv(c, skiprows=9, sep=r'\t', engine='python'))
            
        self.data_list = data_list
        if len(self.data_list) > 1:
            self.df = pd.concat([d.set_index('X') for d in self.data_list], axis=1, join='inner').reset_index()
        else:
            self.df = self.data_list[0]
        self.info_df = pd.read_csv(pathlib.Path(info_file, **kwargs), skiprows=12, sep=r'\t', engine='python')
        
    def plot_mrixs(
        self,
        show=False,
        savefig=None,
        plot_pfy=False,
        pfy_color='gray',
        plot_tfy=True,
        plot_tfy_masked=False,
        tfy_color='black',
        plot_tey=False,
        tey_color='blue',
        izero='Izero',
        drop=None,
        ipfy_lim=None,
        plot_ipfy=False,
        ipfy_color='Blue',
        ipfy_box=False,
        dim=[3.25,3.25],
        xlim=[0,2047],
        header=None,
        tfy_skip=None
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
        Z = (Z-Z.min())/(Z[:,min(xlim):max(xlim)].max()-Z.min())
        # print(Z.max(keepdims=True))
        
        pc = self.axs[0].pcolormesh(x, y, Z, linewidth=0, antialiased=True, alpha=1, edgecolor='face', rasterized=True, vmin=0, vmax=1)
        cbar = self.fig.colorbar(pc)

        # print(self.df.columns)

        # temp_df = self.df.drop(labels='X', axis=1)

        I = Z.sum(axis=1)
        ipfy = np.divide(1, Z[:,min(ipfy_lim):max(ipfy_lim)].sum(axis=1))

        tfy = np.array(self.info_df['TFY'])
        tey = np.array(self.info_df['TEY'])
        
        y_masked = []
        tfy_masked = []
        
        for i in range(len(tfy)):
            if tfy[i] != 8413656:
                y_masked.append(y[i])
                tfy_masked.append(tfy[i])
        
        # print(tfy*(tfy != 8413656))

        if plot_pfy:
            self.axs[1].plot((I-min(I))/(max(I)-min(I)), y, color=pfy_color)
        if plot_tfy:
            self.axs[1].plot((tfy-min(tfy))/(max(tfy)-min(tfy)), y, color=tfy_color)
        if plot_tfy_masked:
            self.axs[1].plot((tfy_masked-min(tfy_masked))/(max(tfy_masked)-min(tfy_masked)), y_masked, color=tfy_color)
        if plot_tey:
            self.axs[1].plot((tey-min(tey))/(max(tey)-min(tey)), y, color=tey_color)
        if plot_ipfy:
            self.axs[1].plot((ipfy-min(ipfy))/(max(ipfy)-min(ipfy)), y, color=ipfy_color)

        self.axs[0].set_xlim(xlim)
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
            a.yaxis.set_minor_locator(MultipleLocator(1))
        self.axs[0].set_xlabel('Emission Energy (a.u.)', fontsize=fontsize)
        self.axs[1].set_xlabel('Norm.\n Intensity', fontsize=fontsize)

        self.axs[0].set_ylabel('Excitation Energy (eV)', fontsize=fontsize)
        
        self.axs[1].set_xticks([])
        self.axs[1].set_yticks([])
        
        self.fig.set_size_inches(*dim)
        if show:
            plt.show()
        if savefig:
            plt.savefig(savefig)
        
        if isinstance(header, str):
            self.axs[0].text(0.025, 0.95, header,
                        verticalalignment='top',
                        horizontalalignment='left',
                        transform=self.axs[0].transAxes,
                        fontsize=fontsize,
                        color='white')
            
    def plot_xes(
        self,
        idx=1,
        fig=None,
        ax=None
    ):
        if fig is None or (ax is None):
            fig, ax = plt.subplots()
        ax.plot(self.df.iloc[:,0], self.df.iloc[:,idx])
        
class Util:
    @staticmethod
    def bulk_data_read(
        dir
    ):
        if isinstance(dir, str):
            dir = pathlib.Path(dir)
        
        dir_list = []
        info_file_list = []
        
        for c in sorted(dir.glob('*')):
            dir_list.append(c/'Andor')
            
        for c in sorted(dir.glob('*/*AI.txt')):
            info_file_list.append(c)
            
        return dir_list, info_file_list
    
    @staticmethod
    def replace_entries(
        rixs0,
        rixs1,
        energy_tol = 0.015,
        energy_col = 'BL 8 Energy'
    ):
        '''
        Replaces data points in Rixs() object rixs0 with corresponding points from rixs1 based on excitation energy.
        
        ARGS:
            rixs0 (pyrixs.Rixs())
            rixs1 (pyrixs.Rixs())
            
        KWARGS:
            energy_tol: Tolerance for matching excitation energies.
        '''
        target_energies = rixs1.info_df[energy_col]
        target_indices = []
        for e in target_energies:
            target_indices.append(Util.find_closest_indices(rixs0.info_df[energy_col], e, energy_tol))

        for i in range(len(target_indices)):
            idx = target_indices[i]
            rixs0.info_df.iloc[idx,:] = rixs1.info_df.iloc[i,:]
            rixs0.info_df["Frame #"].iloc[idx] = float(idx+1)
            rixs0.df.iloc[:,idx+1] = rixs1.df.iloc[:,i+1]
        
    @staticmethod
    def find_closest_indices(
        arr,
        val,
        tol
    ):
        idxs = [ idx for idx,el in enumerate(arr) if (np.abs(el - val) < tol)]
        return idxs[0] if len(idxs) != 0 else np.nan