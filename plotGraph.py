import matplotlib.pyplot as plt
import numpy as np


def multi_plot(rl, em, bleu,interval):
    plt.plot(interval, rl,'+-', label = 'ROUGLE-L')
    plt.plot(interval, em,'+-', label = 'Exact Match')
    plt.plot(interval, bleu,'+-', label = 'BLEU')

    # plt.suptitle('Evaluation with different intervals', fontsize = 15)
    plt.xticks(np.arange(min(interval), max(interval)+1, 10.0))
    plt.legend()
    plt.xlabel('Data Size for Training')
    plt.ylabel('Examination Results (%)')
    # plt.show()
    plt.savefig('vary_training_size.png')


# training ptb on T5-small with different interval
# rl = [88.7157, 96.0398, 97.4281, 98.0455, 98.3347, 98.643, 99.046, 98.9691, 99.3007]
# em = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# bleu = [80.4111, 91.7192, 94.5737, 95.9414, 96.5473, 97.2210, 98.0641, 97.8935, 98.6925]

# training MNLI with different interval
# rl = [84.577,93.0629,94.4945,94.8156,95.0015,94.7083,90.3548,96.9072,90.1408]
# em = [0.0, 0.1686, 0.2551, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# bleu = [72.5129,83.3208,86.2580,86.8868,87.6011,86.6067,83.7559,87.1061,79.2631]
# interval = [50,100,150,200,250,300,350,400,500]

# varing training size to evaluate MNLI
rl = [96.66,96.4118,96.2217,95.7589,95.2565,94.717,94.1502,93.6448,92.53,91.1125]
em = [65.75,56.10,54.225,54.525,47.175,45.45,42.075,41.6,35.375,30.125]
bleu = [92.76,88.6524,88.0968,87.4814,84.4530,83.0427,81.2172,81.9514,77.1580,74.5487]
training_size = [100,90,80,70,60,50,40,30,20,10]

# varing training size to evaluate PTB
# rl = [96.634,97.9235,98.454,98.8243,99.0066,99.2138,99.2944,99.4107,99.5621,99.6295]
# em = [55.3576,69.6889,76.1765,82.0260,84.84445,88.2743,89.3911,91.5979,93.5123,94.6290]
# bleu = [92.1849,95.2080,96.3883,97.4421,97.8715,98.3824,98.5818,98.8738,99.1175,99.3118]
# training_size = [10,20,30,40,50,60,70,80,90,100]

multi_plot(rl,em,bleu,training_size)

