import pyrixs
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd


from matplotlib.ticker import MultipleLocator
from matplotlib import rc, rcParams

dir = '../data/irixs/2024-6-6'
dir_list, info_file_list = pyrixs.Util.bulk_data_read(dir)

ipfy_lim = [0,395]
xlim = [1060, 1250]

name_list = [
    'CoN',
    'CoN3',
    'Co(CO)N',
    'CoCl',
    'CoNCO'
]

# fig, axs = plt.subplots(1,3,layout='constrained')
for i in range(1,len(dir_list)-3):
    rixs = pyrixs.Rixs(dir_list[i], info_file_list[i])
    rixs.plot_mrixs(plot_ipfy=False, ipfy_lim=ipfy_lim, plot_tfy=False, plot_tey=True,
                    xlim=xlim, header=name_list[i-1],
                    cmap='plasma')
plt.show()