import matplotlib.pyplot as plt


def multi_plot(num_epochs, rl, em, bleu,interval):
    epochs = [x+1 for x in range(num_epochs)]
    fig, (ax_rl, ax_em, ax_bleu) = plt.subplots(3)
    for i, inv in enumerate(interval):
        the_label = ''.join([str(inv-50), '-', str(inv)])
        ax_rl.plot(epochs, rl[i], label = the_label)
        ax_em.plot(epochs, em[i], label = the_label)
        ax_bleu.plot(epochs, bleu[i], label = the_label)

    ax_rl.set_xlabel('Epoch number')
    ax_rl.set_ylabel('ROUGE-L SCORE', fontsize = 7)
    ax_rl.set_ylim([0.0, 100.0])
    ax_em.set_xlabel('Epoch number')
    ax_em.set_ylabel('EXACT MATCH SCORE', fontsize = 7)
    ax_em.set_ylim([0.0, 100.0])
    ax_bleu.set_xlabel('Epoch number')
    ax_bleu.set_ylabel('BLEU SCORE', fontsize = 7)
    ax_bleu.set_ylim([0.0, 100.0])
    fig.suptitle('Evaluation with different intervals', fontsize = 15)
    fig.savefig('evaluate.png')


    # epochs = [x+1 for x in range(num_epochs)]
    # fig_rl = plt.figure(figsize=(8, 4))
    # ax_rl = fig_rl.add_subplot(111)
    # for i, inv in enumerate(interval):
    #     the_label = ''.join([str(inv-50), '-', str(inv)])
    #     ax_rl.plot(epochs, rl[i], label = the_label)
    #
    # ax_rl.legend(loc = 0)
    # ax_rl.set_xlabel('Epoch number')
    # ax_rl.set_ylabel('ROUGE-L SCORE')
    # ax_rl.set_ylim([0.0, 100.0])
    # fig_rl.savefig('Rouge-L.png' )
    #
    # fig_em = plt.figure(figsize=(8, 4))
    # ax_em = fig_em.add_subplot(111)
    # for i, inv in enumerate(interval):
    #     the_label = ''.join([str(inv-50), '-', str(inv)])
    #     ax_em.plot(epochs, em[i], label = the_label)
    # ax_em.legend(loc = 0)
    # ax_em.set_xlabel('Epoch number')
    # ax_em.set_ylabel('EXACT MATCH SCORE')
    # ax_em.set_ylim([0.0, 100.0])
    # fig_em.savefig('EM.png' )
    #
    # fig_bleu = plt.figure(figsize=(8, 4))
    # ax_bleu = fig_bleu.add_subplot(111)
    # for i, inv in enumerate(interval):
    #     the_label = ''.join([str(inv-50), '-', str(inv)])
    #     ax_bleu.plot(epochs, bleu[i], label = the_label)
    # ax_bleu.legend(loc = 0)
    # ax_bleu.set_xlabel('Epoch number')
    # ax_bleu.set_ylabel('BLEU SCORE')
    # ax_bleu.set_ylim([0.0, 100.0])
    # fig_bleu.savefig('Bleu.png' )




rl50 = [86.5624,89.5727,89.2914,88.4595,86.5789,90.005,89.4818,89.6489,89.8724,90.2625]
em50 = [23.10,33.10,34.0069,35.0438,33.1294,39.0853,37.8888,38.3408,38.5535,39.3778]
bleu50 = [67.15, 73.817,72.4858,71.1377,65.8279,73.8144,72.0353,72.3936,72.9814,74.0747]

rl100 = [95.5987,96.4129,96.3856,96.6693,97.4641,97.7622,97.1424,97.5203,97.3828,97.3489]
em100 = [52.7253,54.5599,56.5541,56.6604,56.2882,65.4613,66.5514,66.3387,63.8128,62.0845]
bleu100 = [89.8449,91.2500,90.1048,89.6386,92.8025,93.8887,91.1747,92.5739,91.9894,91.4289]

rl150 = [94.4374,97.2629,97.8278,98.0074,98.1207,98.0402,98.0057,97.6072,97.9957,98.0319]
em150 = [49.2422,33.8739,36.1074,30.7630,53.7091,60.8614,53.6293,57.6442,56.6604,55.3310]
bleu150 = [87.4378,92.0135,92.0305,92.7439,94.3736,94.2037,93.7094,92.6884,93.8302,93.8947]

rl200 = [92.8662,91.8344,96.3599,97.4925,97.6170,97.4136,97.5725,97.5635,97.1825,97.303]
em200 = [37.1709,42.4355,45.3868,42.8875,48.5243,45.2007,49.8803,57.4049,49.3485,49.5346]
bleu200 = [87.1414,84.2067,91.7633,93.2513,93.3239,92.7718,93.2991,93.8968,92.2353,92.5614]

rl250 = [85.7675,94.5192,94.9053,94.8032,95.5835,96.1507,96.0054,96.2869,96.3573,96.3793]
em250 = [9.6782,35.0428,41.5580,45.0943,31.1353,40.5211,39.9893,36.4264,38.7131,39.4044]
bleu250 = [73.1558,88.1300,89.7860,90.4170,89.6405,91.1927,91.1299,91.1073,91.3734,91.4607]

rl300 = [25.8894,50.215,70.114,73.7928,75.5229,77.7185,80.2134,80.9068,80.6017,80.6983]
em300 = [0.0531,0.1861,1.1167,1.4623,1.8612,2.9513,3.9883,4.2807,4.1744,4.3073]
bleu300 = [6.6127,25.2860,50.7375,55.4087,57.5489,60.2615,63.3063,64.2970,63.7906,64.0462]

rl350 = [60.7063,37.0565,36.7864,41.7848,46.5283,52.1871,56.2096,59.2226,60.9693,61.5865]
em350 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0797,0.0797,0.0797,0.0797]
bleu350 = [1.2801,4.1616,9.4881,15.0924,20.1032,25.3380,30.1765,34.1648,36.5502,37.4756]

rl400 = [87.0584,80.2194,73.6767,66.637,62.3681,58.7088,56.4598,55.7364,55.2657,55.1991]
em400 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
bleu400 = [0.0340,0.1096,0.3753,0.7066,1.0318,1.3631,1.6650,2.0288,2.2048,2.3077]

rl450 = []
em450 = []
bleu450 = []

rl500 = [88.3406, 83.9537,79.7021,75.0509,70.7303,66.5463,62.8062,60.3186,59.1386,58.4022]
em500 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
bleu500 = [0.0,0.1171,0.2691,0.3822,0.5105,0.5878,0.7563,0.9049,0.9534,1.0012]

# rl = [rl50, rl100, rl150, rl200, rl250, rl300, rl350, rl400, rl450, rl500]
# em = [em50, em100, em150, em200, em250, em300, em350, em400, em450, em500]
# bleu = [bleu50, bleu100, bleu150, bleu200, bleu250, bleu300, bleu350, bleu400, bleu450, bleu500]
# interval = [50,100,150,200,250,300,350,400,450,500]

rl = [rl50, rl100, rl150, rl200, rl250, rl300, rl350, rl400, rl500]
em = [em50, em100, em150, em200, em250, em300, em350, em400, em500]
bleu = [bleu50, bleu100, bleu150, bleu200, bleu250, bleu300, bleu350, bleu400, bleu500]
interval = [50,100,150,200,250,300,350,400,500]


multi_plot(10,rl,em,bleu,interval)