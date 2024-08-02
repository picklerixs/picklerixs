import picklerixs
import matplotlib.pyplot as plt

from scipy.signal import find_peaks

def main():
    dir_stem = '../data/irixs/2024-7-5/CCD Scan 171'
    for i in ['74']:        
        expt_dir = '{}{}'.format(dir_stem, i)
        rixs = picklerixs.Rixs(expt_dir)
        fig, axs = rixs.plot_xes(787.2, xmode='ccd_pixel')
        
        idx, _ = find_peaks(
            rixs.rixs_cut['norm_rixs_intensity'],
            width=1.25,
            prominence=0.00005
        )
        
        axs.plot(idx, rixs.rixs_cut['norm_rixs_intensity'][idx], 'ro')
    plt.show()
    
if __name__ == '__main__':
    main()