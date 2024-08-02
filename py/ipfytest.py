import picklerixs
import matplotlib.pyplot as plt

def main():
    dir_stem = '../data/irixs/2024-7-5/CCD Scan 171'
    expt_dir = '{}{}'.format(dir_stem, '74')
    rixs = picklerixs.Rixs(expt_dir)
    rixs.calc_pfy([1,150], xmode='ccd_pixel')
    # print(rixs.ds['norm_iPFY'])
    # rixs.plot_mrixs(xmode='ccd_pixel', plot_ipfy=True, plot_tey=False, plot_tfy=True)
    rixs.plot_xas(
        plot_tfy=True,
        plot_tey=True,
        plot_ipfy=True,
        offset=0.5
    )
    # rixs.ds.plot.scatter(x='excitation_energy', y='norm_iPFY')
    plt.show()
    
if __name__ == '__main__':
    main()