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


dir = r"../data/irixs/2024-3-28/CCD Scan 16491/Andor"
info_file = r"../data/irixs/2024-3-28/CCD Scan 16491/CoL3_16491-AI.txt"

rixs = pyrixs.Rixs(dir, info_file)
# print(rixs.data_list[0])
print(rixs.info_df)

fig, ax = plt.subplots(1, 2, layout="constrained")
ax[0].plot(rixs.info_df['BL 8 Energy'], rixs.info_df['TFY']/rixs.info_df['Izero'])
# sns.lineplot(rixs.info_df, x='BL 8 Energy', y='TFY', ax=ax)
# sns.lineplot(rixs.info_df, x='BL 8 Energy', y='TEY', ax=ax)

x = rixs.df['X']
y = rixs.info_df['BL 8 Energy']
Z = np.array(rixs.df.iloc[:, 1:]).transpose()

# print(Z)

pc = ax[0].pcolormesh(x, y, Z, linewidth=0, antialiased=True, alpha=1, edgecolor='face', rasterized=True)

# print(rixs.df.columns)

temp_df = rixs.df.drop(labels='X', axis=1)

I = temp_df.sum(axis=0)
print(I)
# print(rixs.df.sum(axis=1))

tfy = rixs.info_df['TFY']
tey = rixs.info_df['TEY']

ax[1].plot((I-min(I))/(max(I)-min(I)), y)
ax[1].plot((tfy-min(tfy))/(max(tfy)-min(tfy)), y)
ax[1].plot((tey-min(tey))/(max(tey)-min(tey)), y)

ax[0].set_xlim([0,2047])
ax[1].set_xlim([-0.1,1.1])
ax[0].set_ylim([775,788])
ax[1].set_ylim([775,788])


for a in ax:
    # a.set_ylabel(ylabel, fontsize=fontsize, labelpad=labelpad*2/3)
    # a.set_xlabel(xlabel, fontsize=fontsize, labelpad=labelpad*2/3)
    a.tick_params(labelsize=fontsize)
    # ax.xaxis.set_tick_params(width=tick_linewidth, length=tick_length, which='major')
    # ax.xaxis.set_tick_params(width=tick_linewidth, length=tick_length*0.5, which='minor')
    # ax.yaxis.set_tick_params(width=tick_linewidth, length=tick_length, which='major')
    # ax.yaxis.set_tick_params(width=tick_linewidth, length=tick_length*0.5, which='minor')
    
ax[0].set_xlabel('Emission Energy (a.u.)', fontsize=fontsize)
ax[1].set_xlabel('Norm. Intensity', fontsize=fontsize)

ax[0].set_ylabel('Excitation Energy (eV)', fontsize=fontsize)
# ax[1].ylabel('Norm. Intensity')

plt.show()