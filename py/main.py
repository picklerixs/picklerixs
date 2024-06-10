import pyrixs
import matplotlib.pyplot as plt
import numpy as np
import pathlib

from matplotlib import rc, rcParams


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


# dir_list = [r"../data/irixs/2024-3-28/CCD Scan 16491/Andor",
#             r"../data/irixs/2024-3-28/CCD Scan 16492/Andor",
#             r"../data/irixs/2024-3-28/CCD Scan 16493/Andor",
#             r"../data/irixs/2024-3-28/CCD Scan 16494/Andor",
#             r"../data/irixs/2024-3-28/CCD Scan 16497/Andor",
#             r"../data/irixs/2024-3-28/CCD Scan 16498/Andor",
#             r"../data/irixs/2024-3-28/CCD Scan 16499/Andor",
#             r"../data/irixs/2024-3-28/CCD Scan 16500/Andor"]
# info_file_list = [r"../data/irixs/2024-3-28/CCD Scan 16491/CoL3_16491-AI.txt",
#              r"../data/irixs/2024-3-28/CCD Scan 16492/CoL3_16492-AI.txt",
#              r"../data/irixs/2024-3-28/CCD Scan 16493/CoL3_16493-AI.txt",
#              r"../data/irixs/2024-3-28/CCD Scan 16494/CoL3_16494-AI.txt",
#              r"../data/irixs/2024-3-28/CCD Scan 16497/CoL3_16497-AI.txt",
#              r"../data/irixs/2024-3-28/CCD Scan 16498/CoL3_16498-AI.txt",
#              r"../data/irixs/2024-3-28/CCD Scan 16499/CoL3_16499-AI.txt",
#              r"../data/irixs/2024-3-28/CCD Scan 16500/CoL3_16500-AI.txt"]

ipfy_lim = [395,800]
xlim = [1525,1675]

header_list = ['CoCl','CoN$_{3}$','CoN','Co(CO)N']

dir = '../data/irixs/2024-3-28'
dir_list, info_file_list = pyrixs.Util.bulk_data_read(dir)

rixs_list = [pyrixs.Rixs(dir_list[i], info_file_list[i]) for i in [2,4,3,5]]

# rixs0 = pyrixs.Rixs(dir_list[2], info_file_list[2])
# rixs1 = pyrixs.Rixs(dir_list[4], info_file_list[4])

pyrixs.Util.replace_entries(rixs_list[0], rixs_list[1])
# pyrixs.Util.replace_entries(rixs_list[2], rixs_list[3])

rixs_list[0].plot_mrixs(show=False, plot_ipfy=True, plot_tfy=False, plot_tfy_masked=True, ipfy_lim=ipfy_lim, dim=[4.5,2.5], xlim=xlim, header='CoN')
# rixs_list[2].plot_mrixs(show=False, plot_ipfy=True, plot_tfy=False, plot_tfy_masked=True, ipfy_lim=ipfy_lim, dim=[4.5,2.5], xlim=xlim, header='Co(CO)N')

rixs_list_new = [rixs_list[0]]
# print(rixs0.df)

# for i in range(len(dir_list)):
for i in [0,1,3]:
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
        drop = [17,19]
    else:
        drop = []
    if i not in [0,1,2,3]:
        header = None
    else:
        header = header_list[i]
    rixs.plot_mrixs(show=False, plot_ipfy=True, ipfy_lim=ipfy_lim, izero=izero, drop=drop, dim=[4.5,2.5], xlim=xlim, header=header)
    rixs_list_new.append(rixs)
        

fontsize=22
labelsize=22
linewidth=3

font_family='Arial'
axes_linewidth=2.25
tick_linewidth=axes_linewidth*.9
tick_length=tick_linewidth*5
marker_size=9
marker_edge_width=linewidth/2
labelpad=15

rc('font',**{'family':'sans-serif','sans-serif':[font_family]})
rc('text', usetex=False)

fig, ax = plt.subplots(layout='constrained')

for rixs in rixs_list_new:
    ax.plot(rixs.y, rixs.ipfy)
ax.legend(['CoN','CoCl','CoN3','Co(CO)N'])

i = 0    
fig, ax = plt.subplots(layout='constrained')
for rixs in rixs_list_new:
    if i == 0:
        ax.plot(rixs.y_masked, rixs.tfy)
    else:
        ax.plot(rixs.y, rixs.tfy)
    i += 1
ax.legend(['CoN','CoCl','CoN3','Co(CO)N'])

# fig, ax = plt.subplots()
# for i in [-2,-1]:
#     dir = dir_list[i]
#     info_file = info_file_list[i]
#     rixs = pyrixs.Rixs(dir, info_file)
#     rixs.plot_xes(fig=fig, ax=ax, idx=1)
    
plt.show()