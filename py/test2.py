import picklerixs
import matplotlib.pyplot as plt

def main():
    i_arr = ['74','75','76','77']
    text_dict = {
            '74': 'CoCl-MFU-4l',
            '75': 'CoN-MFU-4l',
            '76': 'CoN$_{3}$-MFU-4l',
            '77': 'Co(CO)N-MFU-4l'
        }
    dir_stem = '../data/irixs/2024-7-5/CCD Scan 171'
    elastic_xlim = [[1500,1600]]
    elastic_ylim = [[775,778.5], [778.9,780], [783,783.6], [783.8,784.6]]
    # elastic_ylim = [[775,780]]
    for i in ['74']:        
        expt_dir = '{}{}'.format(dir_stem, i)
        rixs = picklerixs.Rixs(expt_dir)
        rixs.find_elastic_line(xlim=elastic_xlim, ylim=elastic_ylim)
        rixs.fit_elastic_line()
        # rixs.plot_mrixs(xmode='emission_energy')
        # rixs.plot_mrixs(xmode='ccd_pixel', plot_elastic_line=True, xlim=elastic_xlim[0])
        #print(rixs.ccd_pixel_arr)
    slope = rixs.slope
    intercept = rixs.intercept
    custom_params = [
        ['slope', slope, True, None, None, None, None],
        ['intercept', intercept, True, None, None, None, None]
    ]
    elastic_ylim = {
        '74': [[775,778.5], [778.9,780], [783,783.6], [783.8,784.6]],
        '75': [[775,778.5], [778.9,780]],
        '76': [[775,778.5], [778.9,779.05], [779.15,799.85], [778.95,780]],
        '77': [[778,782]]
    }
    xlim = [772,799]
    # i_arr = ['75']
    for i in i_arr:        
        expt_dir = '{}{}'.format(dir_stem, i)
        rixs = picklerixs.Rixs(expt_dir)
        rixs.find_elastic_line(
            xlim=elastic_xlim, 
            ylim=elastic_ylim[i]
        )
        rixs.fit_elastic_line(custom_params=custom_params)
        rixs.plot_mrixs(
            xmode='ccd_pixel', 
            plot_elastic_line=True, 
            xlim=elastic_xlim[0],
            text=text_dict[i]
        )
        # rixs.plot_mrixs(dim=[3.75,2.75], text=text_dict[i], xmode='emission_energy', xmajtm=50, xmintm=10, ymajtm=5, ymintm=1, xlim=xlim, savefig='CoL3_hrRIXS_full_range{}.png'.format(i))
        #rixs.plot_mrixs(dim=[3.75,2.75], text=text_dict[i], xmode='emission_energy', xmajtm=10, xmintm=2, ymajtm=5, ymintm=1, xlim=[765,795], savefig='CoL3_zoomed_more{}.png'.format(i))
        #rixs.plot_mrixs(xmode='ccd_pixel', plot_elastic_line=True)
        print(rixs.ccd_pixel_arr)
    plt.show()

if __name__ == '__main__':
    main()
