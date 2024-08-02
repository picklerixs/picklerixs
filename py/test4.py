import picklerixs
import matplotlib.pyplot as plt

def main():
    i_arr = ['74','75','76','77']
    text_dict = {
            '74': 'CoN-MFU-4l',
            '75': 'CoN$_{3}$-MFU-4l',
            '76': 'Co(CO)N-MFU-4l',
            '77': 'CoCl-MFU-4l',
        }
    # xlim = [[1000,1500]]
    dir_stem = '../data/irixs/2024-7-5/CCD Scan 171'
    # for i in ['11']:        
    #     expt_dir = '{}{}'.format(dir_stem, i)
    #     rixs = picklerixs.Rixs(expt_dir)
    #     rixs.find_elastic_line(
    #         xlim=xlim,
    #         width=None,
    #         distance=9999
    #     )
    #     rixs.fit_elastic_line()
    #     rixs.plot_mrixs(xmode='emission_energy')
    #     rixs.plot_mrixs(
    #         xmode='ccd_pixel', 
    #         plot_elastic_line=True,
    #         dim=[5.5,4.5],
    #         savefig='CoL3_ccd_elastic_test{}.svg'.format(i),
    #         xlim=[1000,1400]
    #     )
    #     #print(rixs.ccd_pixel_arr)
    # slope = rixs.slope
    # custom_params = [['slope', slope, False, None, None, None, None]]
    for i in i_arr:        
        expt_dir = '{}{}'.format(dir_stem, i)
        rixs = picklerixs.Rixs(expt_dir)
        rixs.calc_pfy([1,150], xmode='ccd_pixel')
        # rixs.find_elastic_line(xlim=xlim, width=None, distance=9999)
        # rixs.fit_elastic_line(custom_params=custom_params)
        rixs.plot_mrixs(
            dim=[3.75,2.35], 
            text=text_dict[i], 
            xmode='ccd_pixel', 
            # xlim=[765,795], 
            xlim=[1480,1580],
            ylim=[775, 788],
            xlabel='CCD Pixel',
            ymajtm=5, ymintm=1,
            plot_ipfy=True,
            plot_tey=False,
            plot_tfy=False,
            # xmajtm=10, xmintm=2, ymajtm=5, ymintm=1,
            savefig='CoL3_full_range_hrRIXS{}.svg'.format(i)
        )
        rixs.plot_xas(
            dim=[3.75,2.35], 
            text=text_dict[i], 
            plot_tfy=True,
            plot_tey=True,
            plot_ipfy=True,
            offset=0.5,
            xlim=[775, 788],
            xmajtm=5, xmintm=1,
            savefig='CoL3_NEXAFS{}.svg'.format(i)
        )
        # rixs.plot_mrixs(dim=[3.75,2.75], text=text_dict[i], xmode='emission_energy', xmajtm=10, xmintm=2, ymajtm=5, ymintm=1, xlim=[765,795], savefig='CoL3_zoomed_more{}.png'.format(i))
        # rixs.plot_mrixs(xmode='ccd_pixel', plot_elastic_line=True)
        # print(rixs.ccd_pixel_arr)
    # plt.show()

if __name__ == '__main__':
    main()
