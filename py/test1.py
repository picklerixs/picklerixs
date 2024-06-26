import picklerixs
import matplotlib.pyplot as plt

def main():
    i_arr = ['09','11','10','12']
    for i in i_arr:        
        expt_dir = '../data/irixs/2024-6-6/CCD Scan 70{}'.format(i)
        rixs = picklerixs.Rixs(expt_dir)
        rixs.find_elastic_line(xlim=[1000,1500])
        rixs.fit_elastic_line()
        rixs.plot_mrixs(xmode='ccd_pixel', plot_elastic_line=True)
        print(rixs.ccd_pixel_arr)
    plt.show()

if __name__ == '__main__':
    main()
