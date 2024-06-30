import picklerixs
import matplotlib.pyplot as plt

def main():
    i_arr = ['09','11','10','12']
    for i in ['11']:        
        expt_dir = '../data/irixs/2024-6-6/CCD Scan 70{}'.format(i)
        rixs = picklerixs.Rixs(expt_dir)
        rixs.find_elastic_line(xlim=[1000,1500])
        rixs.fit_elastic_line()
        #rixs.plot_mrixs(xmode='emission_energy')
        #rixs.plot_mrixs(xmode='ccd_pixel', plot_elastic_line=True)
        #print(rixs.ccd_pixel_arr)
    slope = rixs.slope
    custom_params = [['slope', slope, False, None, None, None, None]]
    for i in i_arr:        
        expt_dir = '../data/irixs/2024-6-6/CCD Scan 70{}'.format(i)
        rixs = picklerixs.Rixs(expt_dir)
        rixs.find_elastic_line(xlim=[1000,1500])
        rixs.fit_elastic_line(custom_params=custom_params)
        rixs.plot_mrixs(dim=[3.75,2.75], xmode='emission_energy', xlim=[765,795], savefig='CoL3_zoomed_more{}.png'.format(i))
        #rixs.plot_mrixs(xmode='ccd_pixel', plot_elastic_line=True)
        print(rixs.ccd_pixel_arr)
    #plt.show()

if __name__ == '__main__':
    main()
