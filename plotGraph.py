import matplotlib.pyplot as plt
import numpy as np


def multi_plot(rl, em, bleu,interval):
    plt.plot(interval, rl,'+-', label = 'ROUGLE-L')
    plt.plot(interval, em,'+-', label = 'Exact Match')
    plt.plot(interval, bleu,'+-', label = 'BLEU')

    # plt.bar(interval, rl,10,color='#FBDD7E')
    # plt.bar(interval, em, 10, bottom=rl,color='#04D8B2')
    # plt.bar(interval, bleu, 10, bottom=rl+em,color='#7BC8F6')

    # plt.suptitle('Evaluation with different intervals', fontsize = 15)
    my_xticks = ['<=100','100 ~ 200','>200']
    plt.xticks(interval,my_xticks)
    # plt.ylim(88,100)
    plt.margins(x=0)
    # plt.legend(['ROUGLE-L','Exact Match','BLEU'])
    plt.legend()
    plt.xlabel('Various Length interval')
    plt.ylabel('Examination Results (%)')
    # plt.show()
    plt.savefig('evaluate_out.png')


# training ptb on T5-small with different interval
# rl = np.array([99.4167,99.736,99.6986])
# em = np.array([96.6470,92.7616,89.0566])
# bleu = np.array([99.2728,99.2393,99.2781])

rl = []
em = []
bleu = []


# training MNLI with different interval
# rl = np.array([96.0511,96.5601,96.0815])
# em = np.array([57.9411,32.6837,13.5849])
# bleu = np.array([89.9814,89.9058,89.2588])

rl = [95.1088,96.6674,96.8095,96.4849,96.3387]
em = [70.4285,51.5193,41.9354,33.1230,21.62764]
bleu = [90.1946,90.1336,89.9898,89.7656,89.4585]

interval = [60,90,120,150,500]

# varing training size to evaluate MNLI
# rl = [96.66,96.4118,96.2217,95.7589,95.2565,94.717,94.1502,93.6448,92.53,91.1125]
# em = [65.75,56.10,54.225,54.525,47.175,45.45,42.075,41.6,35.375,30.125]
# bleu = [92.76,88.6524,88.0968,87.4814,84.4530,83.0427,81.2172,81.9514,77.1580,74.5487]
# training_size = [100,90,80,70,60,50,40,30,20,10]

# varing training size to evaluate PTB
# rl = [96.634,97.9235,98.454,98.8243,99.0066,99.2138,99.2944,99.4107,99.5621,99.6295]
# em = [55.3576,69.6889,76.1765,82.0260,84.84445,88.2743,89.3911,91.5979,93.5123,94.6290]
# bleu = [92.1849,95.2080,96.3883,97.4421,97.8715,98.3824,98.5818,98.8738,99.1175,99.3118]
# training_size = [10,20,30,40,50,60,70,80,90,100]

# multi_plot(rl,em,bleu,training_size)
multi_plot(rl,em,bleu,interval)
