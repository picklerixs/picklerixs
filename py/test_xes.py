import picklerixs
import matplotlib.pyplot as plt

'''
Test Rixs.plot_xes()
'''

def main():
    i_arr = ['74','75','76','77']
    text_dict = {
            '74': 'CoN-MFU-4l',
            '75': 'CoN$_{3}$-MFU-4l',
            '76': 'Co(CO)N-MFU-4l',
            '77': 'CoCl-MFU-4l',
        }
    i_arr = ['74']
    dir_stem = '../data/irixs/2024-7-5/CCD Scan 171'
    for i in i_arr:        
        expt_dir = '{}{}'.format(dir_stem, i)
        rixs = picklerixs.Rixs(expt_dir)
        rixs.calc_pfy([1,150], xmode='ccd_pixel')
        rixs.plot_xes([i for i in range(775,788,2)], 
                      xmode='ccd_pixel',
                      offset=0.0005
        )
    plt.show()
    
if __name__ == '__main__':
    main()