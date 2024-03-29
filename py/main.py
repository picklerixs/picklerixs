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
            r"../data/irixs/2024-3-28/CCD Scan 16492/Andor",
            r"../data/irixs/2024-3-28/CCD Scan 16493/Andor",
            r"../data/irixs/2024-3-28/CCD Scan 16494/Andor",
            r"../data/irixs/2024-3-28/CCD Scan 16497/Andor",
            r"../data/irixs/2024-3-28/CCD Scan 16498/Andor",
            r"../data/irixs/2024-3-28/CCD Scan 16499/Andor",
            r"../data/irixs/2024-3-28/CCD Scan 16500/Andor"]
info_file_list = [r"../data/irixs/2024-3-28/CCD Scan 16491/CoL3_16491-AI.txt",
             r"../data/irixs/2024-3-28/CCD Scan 16492/CoL3_16492-AI.txt",
             r"../data/irixs/2024-3-28/CCD Scan 16493/CoL3_16493-AI.txt",
             r"../data/irixs/2024-3-28/CCD Scan 16494/CoL3_16494-AI.txt",
             r"../data/irixs/2024-3-28/CCD Scan 16497/CoL3_16497-AI.txt",
             r"../data/irixs/2024-3-28/CCD Scan 16498/CoL3_16498-AI.txt",
             r"../data/irixs/2024-3-28/CCD Scan 16499/CoL3_16499-AI.txt",
             r"../data/irixs/2024-3-28/CCD Scan 16500/CoL3_16500-AI.txt"]

ipfy_lim = [395,800]

# for i in range(len(dir_list)):
for i in [0,1,2,3]:
    dir = dir_list[i]
    info_file = info_file_list[i]
    rixs = pyrixs.Rixs(dir, info_file)
    # if i == 2:
    #     izero = 'Izero 2'
    # else:
    #     izero = 'Izero'
    # if i == 2:
    #     drop = [17,19,20]
    izero = 'Izero'
    if i == 2:
        # drop = [17,19]
        drop = []
    else:
        drop = []
    rixs.plot_mrixs(show=False, plot_ipfy=True, ipfy_lim=ipfy_lim, izero=izero, drop=drop,dim=[4,2])

# fig, ax = plt.subplots()
# for i in [0]:
#     dir = dir_list[i]
#     info_file = info_file_list[i]
#     rixs = pyrixs.Rixs(dir, info_file)
#     rixs.plot_xes(fig=fig, ax=ax, idx=1)
    
plt.show()