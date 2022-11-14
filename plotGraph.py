import matplotlib.pyplot as plt
import numpy as np


def multi_plot(rl, em, bleu,interval):
    plt.plot(interval, rl, label = 'ROUGLE-L')
    plt.plot(interval, em, label = 'Exact Match')
    plt.plot(interval, bleu, label = 'BLEU')

    # plt.suptitle('Evaluation with different intervals', fontsize = 15)
    plt.xticks(np.arange(min(interval), max(interval)+1, 10.0))
    plt.legend()
    # plt.show()
    plt.savefig('vary_training_size.png')


# training ptb with different interval
# rl = [88.7157, 96.0398, 97.4281, 98.0455, 98.3347, 98.643, 99.046, 98.9691, 99.3007]
# em = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# bleu = [80.4111, 91.7192, 94.5737, 95.9414, 96.5473, 97.2210, 98.0641, 97.8935, 98.6925]

# training MNLI with different interval
# rl = [84.577,93.0629,94.4945,94.8156,95.0015,94.7083,90.3548,96.9072,90.1408]
# em = [0.0, 0.1686, 0.2551, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# bleu = [72.5129,83.3208,86.2580,86.8868,87.6011,86.6067,83.7559,87.1061,79.2631]
# interval = [50,100,150,200,250,300,350,400,500]

# varing training size to evaluate
rl = [98.1026,97.3906,97.0976,96.66,96.1379,95.9289,95.0885,94.1745,93.8181,91.685]
em = [77.65,66.0,63.9,65.75,54.30,55.45,48.75,36.325,40.95,34.525]
bleu = [95.4412,92.0198,91.5778,92.76,87.9033,88.4148,84.5863,75.7713,81.2965,77.71]
training_size = [100,90,80,70,60,50,40,30,20,10]

multi_plot(rl,em,bleu,training_size)

