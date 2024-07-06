import picklerixs
import matplotlib.pyplot as plt

def main():
    i_arr = ['74','75','76']
    text_dict = {
            '74': 'CoCl-MFU-4l',
            '75': 'CoN-MFU-4l',
            '76': 'CoN$_{3}$-MFU-4l',
        }
    dir_stem = '../data/irixs/2024-7-5/CCD Scan 171'
    elastic_xlim = [1400,1600]
    elastic_ylim = [775,778]
    for i in ['74']:        
        expt_dir = '{}{}'.format(dir_stem, i)
        rixs = picklerixs.Rixs(expt_dir)
        rixs.find_elastic_line(xlim=elastic_xlim, ylim=elastic_ylim)
        rixs.fit_elastic_line()
        rixs.plot_mrixs(xmode='emission_energy')
        rixs.plot_mrixs(xmode='ccd_pixel', plot_elastic_line=True)
        #print(rixs.ccd_pixel_arr)
    slope = rixs.slope
    custom_params = [['slope', slope, False, None, None, None, None]]
    xlim = None
    for i in i_arr:        
        expt_dir = '{}{}'.format(dir_stem, i)
        rixs = picklerixs.Rixs(expt_dir)
        rixs.find_elastic_line(xlim=elastic_xlim, ylim=elastic_ylim)
        rixs.fit_elastic_line(custom_params=custom_params)
        rixs.plot_mrixs(dim=[3.75,2.75], text=text_dict[i], xmode='ccd_pixel', xmajtm=50, xmintm=10, ymajtm=5, ymintm=1, xlim=xlim, savefig='CoL3_hrRIXS_full_range{}.png'.format(i))
        #rixs.plot_mrixs(dim=[3.75,2.75], text=text_dict[i], xmode='emission_energy', xmajtm=10, xmintm=2, ymajtm=5, ymintm=1, xlim=[765,795], savefig='CoL3_zoomed_more{}.png'.format(i))
        #rixs.plot_mrixs(xmode='ccd_pixel', plot_elastic_line=True)
        print(rixs.ccd_pixel_arr)
    plt.show()

if __name__ == '__main__':
    main()
