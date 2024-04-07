import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pathlib
import warnings

from scipy.integrate import trapezoid
from scipy.stats import linregress
from scipy.signal import savgol_filter
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
        calibration_data=None,
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
        
        self.calibration_data = np.array(calibration_data)
        
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
        tfy_skip=None,
        x_minor_tick_multiple=1,
        y_major_tick_multiple=4,
        y_minor_tick_multiple=1,
        vmin=0,
        vmax=1
    ):
        fontsize=12
        font_family='Arial'
        axes_linewidth=2.25
        tick_linewidth=axes_linewidth*.9
        tick_length=tick_linewidth*5
        
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
        if self.calibration_data is not None:
            x = self.calibration_data
        else:
            x = np.array(self.df['X'])
        y = np.array(self.info_df['BL 8 Energy'])
        Z = np.array(self.df.iloc[:, 1:]).transpose()
        
        # print(I0)
        
        for i in range(len(I0)):
            Z[:,i] = Z[:,i]/I0[i]
            
        # print(Z.argmax())
        idxmin = abs(x - xlim[0]).argmin()
        idxmax = abs(x - xlim[1]).argmin()
        Z = (Z-Z.min())/(Z[:,idxmin:idxmax].max()-Z.min())
        # print(Z.max(keepdims=True))
        
        pc = self.axs[0].pcolormesh(x, y, Z, linewidth=0, antialiased=True, alpha=1, edgecolor='face', rasterized=True, vmin=vmin, vmax=vmax)
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
            # a.xaxis.set_tick_params(width=tick_linewidth, length=tick_length, which='major')
            # a.xaxis.set_tick_params(width=tick_linewidth, length=tick_length*0.5, which='minor')
            # a.yaxis.set_tick_params(width=tick_linewidth, length=tick_length, which='major')
            # a.yaxis.set_tick_params(width=tick_linewidth, length=tick_length*0.5, which='minor')
            
            a.yaxis.set_major_locator(MultipleLocator(y_major_tick_multiple))
            a.yaxis.set_minor_locator(MultipleLocator(y_minor_tick_multiple))
        self.axs[0].set_xlabel('Emission Energy (eV)', fontsize=fontsize)
        self.axs[1].set_xlabel('Norm.\n Intensity', fontsize=fontsize)

        self.axs[0].set_ylabel('Excitation Energy (eV)', fontsize=fontsize)
        
        self.axs[1].set_xticks([])
        self.axs[1].set_yticks([])
        
        self.axs[0].xaxis.set_minor_locator(MultipleLocator(x_minor_tick_multiple))
        
        self.fig.set_size_inches(*dim)
        if show:
            plt.show()
        
        if isinstance(header, str):
            self.axs[0].text(0.025, 0.95, header,
                        verticalalignment='top',
                        horizontalalignment='left',
                        transform=self.axs[0].transAxes,
                        fontsize=fontsize,
                        color='white')
        
        if plot_tfy_masked:
            self.tfy = (tfy_masked-min(tfy_masked))/(max(tfy_masked)-min(tfy_masked))
            self.y_masked = y_masked
        else:
            self.tfy = (tfy-min(tfy))/(max(tfy)-min(tfy))
        self.ipfy = (ipfy-min(ipfy))/(max(ipfy)-min(ipfy))
        self.y = y
        if savefig:
            plt.savefig(savefig)
            
    def plot_xes(
        self,
        idx=1,
        fig=None,
        ax=None,
        xlim=None,
        ylim=None,
        color=None,
        offset=0,
        dim=[3.25,3.25],
        x_minor_tick_multiple=None,
        x_major_tick_multiple=None,
        savefig=None,
        filter=None,
        filter_args=None,
        filter_kwargs=None,
        norm='minmax',
        norm_kwargs=None,
        # fontsize=12,
        apply_plot_opts=True,
        plot_opts_kwargs={
            'xlabel': 'Emission Energy (eV)'
        }
    ):
        
        if fig is None or (ax is None):
            fig, ax = plt.subplots(layout='constrained')
        if self.calibration_data is not None:
            x = self.calibration_data
        else:
            x = self.df.iloc[:,0]
            
        if filter_args is None:
            filter_args = []
        if filter_kwargs is None:
            filter_kwargs = {}
            
        if filter:
            y = savgol_filter(self.df.iloc[:,idx], *filter_args, **filter_kwargs)
        else:
            y = self.df.iloc[:,idx]
            
        if norm == 'minmax':
            y = (y-min(y))/(max(y)-min(y))
        elif norm == 'area':
            if norm_kwargs is None:
                norm_kwargs = {}
            _, _, _, area = Util.integrate(
                x,
                y,
                xlim,
                **norm_kwargs
            )
            y = y/area
            
        ax.plot(x, y+offset, color=color)
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        # ax.tick_params(labelsize=fontsize)
            
        # ax.set_xlabel('Emission Energy (eV)', fontsize=fontsize)
        # ax.set_ylabel('Intensity (a.u.)', fontsize=fontsize)
        # ax.set_yticks([])
        # if x_minor_tick_multiple:
        #     ax.xaxis.set_minor_locator(MultipleLocator(x_minor_tick_multiple))
        # if x_major_tick_multiple:
        #     ax.xaxis.set_major_locator(MultipleLocator(x_major_tick_multiple))
        
        fig.set_size_inches(*dim)
        
        if plot_opts_kwargs is None:
            plot_opts_kwargs = {}
        if apply_plot_opts:
            Util.plot_opts(fig, ax, **plot_opts_kwargs)
        
        if savefig:
            fig.savefig(savefig)
        return fig, ax
            
            
class Xas:
    # TODO read csv header to JSON/dict
    def __init__(
        self,
        spec_dir,
        id,
        prefix='SigScan',
        calibration_data=None,
        skiprows=14,
        **kwargs
    ):
        self.spec_dir = pathlib.Path(spec_dir, **kwargs)
        data_list = []
        self.id_list = ['{}{}'.format(prefix, x) for x in id]

        self.child_list = self.spec_dir.glob('{}*.txt'.format(prefix))
        self.child_list = sorted(self.child_list)
        self.child_list = [x for x in self.child_list if x.stem in self.id_list]
        for c in self.child_list:
            data_list.append(pd.read_csv(c, skiprows=skiprows, sep=r'\t', engine='python'))
            
        self.data_list = data_list
        if len(self.data_list) > 1:
            self.df = reduce(lambda df0, df1: df0.combine(df1, lambda x0, x1: np.average([x0, x1], axis=0)), self.data_list)
            # self.df = pd.concat([d.set_index('X') for d in self.data_list], axis=1, join='inner').reset_index()
        else:
            self.df = self.data_list[0]
        # self.info_df = pd.read_csv(pathlib.Path(info_file, **kwargs), skiprows=12, sep=r'\t', engine='python')
        
    def plot(
        self,
        fig=None,
        ax=None,
        x='Mono Energy',
        y='Counter 2',
        i0='Counter 0',
        apply_plot_opts=True,
        norm='minmax',
        plot_opts_kwargs=None,
        offset=0,
        **kwargs
    ):
        if y == 'TEY':
            y = 'Counter 1'
        elif y == 'TFY':
            y = 'Counter 2'
        if not ax:
            fig, ax = plt.subplots(layout='constrained')
        y_arr = self.df[y]/self.df[i0]
        if norm:
            y_arr = (y_arr-min(y_arr))/(max(y_arr)-min(y_arr))
        y_arr += offset
        ax.plot(self.df['Mono Energy'], y_arr, **kwargs)
        
        if not plot_opts_kwargs:
            plot_opts_kwargs = {}
        
        if apply_plot_opts:
            Util.plot_opts(fig, ax, **plot_opts_kwargs)
            
        self.y_arr = y_arr
        
        return fig, ax
            
        
class Util:
    @staticmethod
    def bulk_data_read(
        dir,
        data_dir="Andor",
        info_file_suffix="AI.txt"
    ):
        if isinstance(dir, str):
            dir = pathlib.Path(dir)
        
        dir_list = []
        info_file_list = []
        
        for c in sorted([x for x in dir.iterdir() if x.is_dir()]):
            dir_list.append(c/data_dir)
            
        for c in sorted(dir.glob('*/*{}'.format(info_file_suffix))):
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
    
    @staticmethod
    def plot_opts(
        fig,
        ax,
        dim=[3.25,3.25],
        fontsize=12,
        xmintm=1,
        xmajtm=5,
        xlabel='Excitation Energy (eV)',
        ylabel='Intensity (a.u.)',
        xticks=None,
        yticks=[],
        axis_linewidth=None,
        linewidth=None
    ):
        fig.set_size_inches(*dim)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if xmintm:
            ax.xaxis.set_minor_locator(MultipleLocator(xmintm))
        if xmajtm:
            ax.xaxis.set_major_locator(MultipleLocator(xmajtm))
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.tick_params(labelsize=fontsize)
        
        if axis_linewidth:
            for axis in ['top','bottom','left','right']:
                ax.spines[axis].set_linewidth(axis_linewidth)
            tick_linewidth = axis_linewidth*0.9
            tick_length = tick_linewidth*5
            ax.tick_params(width=tick_linewidth, which='both')
            ax.xaxis.set_tick_params(length=tick_length, which='major')
            ax.xaxis.set_tick_params(length=tick_length*0.5, which='minor')
            
        if linewidth:
            for line in ax.lines:
                line.set_linewidth(linewidth)
        
    # TODO add logic to input idxmin, idxmax directly instead of searching for xlim if desired
    @staticmethod
    def integrate(
        x,
        y,
        xlim,
        background='linear',
        background_points=[4,4]
    ):
        idxmin = abs(x - xlim[0]).argmin()
        idxmax = abs(x - xlim[1]).argmin()
        
        x = np.array(x)
        y = np.array(y)
        
        x = x[idxmin:idxmax]
        y = y[idxmin:idxmax]
        
        # x = x.iloc[idxmin:idxmax]
        # y = y.iloc[idxmin:idxmax]
        
        if background == 'linear':
            background_res = linregress(
                np.concatenate([x[:background_points[0]], x[-background_points[1]:]]),
                np.concatenate([y[:background_points[0]], y[-background_points[1]:]])
            )
            y_background = background_res.slope*x + background_res.intercept
            y_no_background = y - y_background
        else:
            y_no_background = y
            
        area = trapezoid(y_no_background, x=x)
            
        return x, y, y_background, area