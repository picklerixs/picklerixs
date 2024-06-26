import pyrixs
import pathlib

import matplotlib.pyplot as plt

def main():
    xas_list = []
    spec_dir = '../data/xas/2024-6-6'
    id_list = [[74974,74980], [74978,74979]]

    fig, ax = plt.subplots(layout='constrained')
    for i in range(len(id_list)):
        xas_list.append(pyrixs.Xas(spec_dir, id_list[i], skiprows=15))
        xas_list[i].plot(y='TEY', fig=fig, ax=ax)
    plt.show()

if __name__ == '__main__':
    main()
