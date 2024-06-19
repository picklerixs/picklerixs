import pyrixs
import matplotlib.pyplot as plt

def main():
    for i in ['09','11','10','12']:        
        expt_dir = '../data/irixs/2024-6-6/CCD Scan 70{}'.format(i)
        rixs = pyrixs.Rixs(expt_dir)
        rixs.plot_mrixs()
    plt.show()

if __name__ == '__main__':
    main()
