import matplotlib.pyplot as plt
import numpy as np


def multi_plot(rl, em, bleu,interval):
    plt.plot(interval, rl, label = 'ROUGLE-L')
    plt.plot(interval, em, label = 'Exact Match')
    plt.plot(interval, bleu, label = 'BLEU')

    plt.suptitle('Evaluation with different intervals', fontsize = 15)
    plt.xticks(np.arange(min(interval), max(interval)+1, 50.0))
    plt.legend()
    # plt.show()
    plt.savefig('evaluate.png')



rl = [88.7157, 96.0398, 97.4281, 98.0455, 98.3347, 98.643, 99.046, 98.9691, 99.3007]
em = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bleu = [80.4111, 91.7192, 94.5737, 95.9414, 96.5473, 97.2210, 98.0641, 97.8935, 98.6925]
interval = [50,100,150,200,250,300,350,400,500]


multi_plot(rl,em,bleu,interval)