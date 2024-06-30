import picklerixs
import matplotlib.pyplot as plt

def main():
    i_arr = ['09','11','10','12','13']
    text_dict = {
            '09': 'CoN-MFU-4l',
            '10': 'CoN$_{3}$-MFU-4l',
            '11': 'Co(CO)N-MFU-4l',
            '12': 'CoCl-MFU-4l',
            '13': 'CoNCO-MFU-4l'
        }
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
        rixs.plot_mrixs(dim=[3.75,2.75], text=text_dict[i], xmode='emission_energy', xmajtm=50, xmintm=10, ymajtm=5, ymintm=1, xlim=[600,850], savefig='CoL3_full_range{}.png'.format(i))
        #rixs.plot_mrixs(dim=[3.75,2.75], text=text_dict[i], xmode='emission_energy', xmajtm=10, xmintm=2, ymajtm=5, ymintm=1, xlim=[765,795], savefig='CoL3_zoomed_more{}.png'.format(i))
        #rixs.plot_mrixs(xmode='ccd_pixel', plot_elastic_line=True)
        print(rixs.ccd_pixel_arr)
    #plt.show()

if __name__ == '__main__':
    main()
