import pyrixs
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


major_tick_multiple = 5
minor_tick_multiple = 1
ylabel = 'Temperature (°C)'
xlabel = 'Binding Energy (eV)'
fontsize = 12
labelpad = 5

font_family='Arial'
axes_linewidth=2.25
tick_linewidth=axes_linewidth*.9
tick_length=tick_linewidth*5


dir_list = [r"../data/irixs/2024-3-28/CCD Scan 16491/Andor",
            r"../data/irixs/2024-3-28/CCD Scan 16492/Andor"]
info_file_list = [r"../data/irixs/2024-3-28/CCD Scan 16491/CoL3_16491-AI.txt",
             r"../data/irixs/2024-3-28/CCD Scan 16492/CoL3_16492-AI.txt"]

for i in range(len(dir_list)):
    dir = dir_list[i]
    info_file = info_file_list[i]
    rixs = pyrixs.Rixs(dir, info_file)
    rixs.plot_mrixs(show=False, plot_tey=False)
    
plt.show()