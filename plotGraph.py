import matplotlib.pyplot as plt
import numpy as np


def multi_plot(rl, em, bleu,interval):
    plt.plot(interval, rl, label = 'ROUGLE-L')
    plt.plot(interval, em, label = 'Exact Match')
    plt.plot(interval, bleu, label = 'BLEU')

    # plt.suptitle('Evaluation with different intervals', fontsize = 15)
    plt.xticks(np.arange(min(interval), max(interval)+1, 50.0))
    plt.legend()
    # plt.show()
    plt.savefig('evaluate_in.png')


# training ptb with different interval
# rl = [88.7157, 96.0398, 97.4281, 98.0455, 98.3347, 98.643, 99.046, 98.9691, 99.3007]
# em = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# bleu = [80.4111, 91.7192, 94.5737, 95.9414, 96.5473, 97.2210, 98.0641, 97.8935, 98.6925]

# training MNLI with different interval
# rl = [84.577,93.0629,94.4945,94.8156,95.0015,94.7083,90.3548,96.9072,90.1408]
# em = [0.0, 0.1686, 0.2551, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
# bleu = [72.5129,83.3208,86.2580,86.8868,87.6011,86.6067,83.7559,87.1061,79.2631]

# varing training size to evaluate
rl = [96.6494]
em = [65.75]
bleu = [92.7664]

interval = [50,100,150,200,250,300,350,400,500]

multi_plot(rl,em,bleu,interval)