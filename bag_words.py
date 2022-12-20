import pandas as pd
import re
import spacy
from collections import Counter
from collections import defaultdict

def extract_data(path):
    sentence_list = []
    lines = []
    with open(path) as file_in:
        for line in file_in:
            # print(line)
            preprocessed = line.replace('<unk>','').replace('\n','')
            # print(preprocessed)
            # preprocessed = re.split(r'\s{3,}',line.replace('\t', '   ').replace('\n', ''))
            lines.append(preprocessed)
            # sentence_list.append(preprocessed[5])
            # sentence_list.append(preprocessed[6])
    # print(len(sentence_list))
    # return sentence_list
    return lines

# match sentence with the corresponding bag of words

def give_bags_words(sentences):
    # sen_with_words = defaultdict(list)
    nlp = spacy.load("en_core_web_sm")
    count = 1
    corresponding_words = []
    for one in sentences:
        doc = nlp(one)
        corresponding_words.append([token.text for token in doc])
        count += 1
        # print(count)
    # print("sentences length is ",len(sentences))
    # print("words length is",len(corresponding_words))
    data={'source_text':corresponding_words,'target_text':sentences}
    df = pd.DataFrame(data)
    df.to_csv('ptb_500.csv',index = False)
    return df

# Count words occurrences

# def give_count_words(sentences):
#     # sen_with_words = defaultdict(list)
#     nlp = spacy.load("en_core_web_sm")
#     count = 1
#     corresponding_words = []
#     for one in sentences:
#         print(one)
#         doc = nlp(one)
#         item_list = [token.text for token in doc]
#         counts = Counter(item_list)
#         counts_list = [(k, v) for k, v in counts.items()]
#         count += 1
#         print(count)
#         print(counts_list)
#         corresponding_words.append(counts_list)
#     # print("sentences length is ",len(sentences))
#     # print("words length is",len(corresponding_words))
#     data={'Sentences':sentences,'Corresponding_words':corresponding_words}
#     df = pd.DataFrame(data)
#     df.to_csv('count_test_ptb.csv')
#     return df


def len_interval(list, min, max):
    # a_list = [a for a in list if len(a) <= max and len(a) > min]
    a_list  = []
    for a in list:
        # print('1')
        if len(a) <= max and len(a) > min:
            a_list.append(a)
            print('aaa: ', a)
            print('bbb: ', len(a))
    print(len(a_list))
    return a_list

def count_avg_len(list):
    all_sen = [len(a) for a in list]
    # print(all_sen)
    avg = sum(all_sen)/len(all_sen)
    return avg
# path = '/Users/dyt/Documents/WORK/Yin/multinli_1.0/multinli_1.0_dev_matched.txt'
# path = '/Users/dyt/Documents/WORK/Yin/ptbdataset/ptb.train.txt'
path = '/Users/dyt/Documents/WORK/Yin/ptbdataset/ptb.test.txt'

sorted_list = extract_data(path)
# df = give_bags_words(sentence_list)
# sorted_list = sorted(sentence_list, key = len)
# print(len(sorted_list))
# print(count_avg_len(sorted_list))

# list_50 = len_interval(sorted_list,0, 50) #506
# list_100 = len_interval(sorted_list,50, 100) #1194
# list_150 = len_interval(sorted_list,100, 150) #1164
# list_200 = len_interval(sorted_list,150, 200) #632
# list_250 = len_interval(sorted_list,200, 250) #218
# list_300 = len_interval(sorted_list,250, 300) #36
# list_350 = len_interval(sorted_list,300, 350) #9
# list_400 = len_interval(sorted_list,350, 400)#1
# list_450 = len_interval(sorted_list,400, 450)#0
list_500 = len_interval(sorted_list,450, 500) #1
give_bags_words(list_500)

# df = give_bags_words(sentence_list)
# df = give_count_words(sentence_list)