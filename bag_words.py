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
            preprocessed = line.replace('<unk>','')
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
        print(count)
    # print("sentences length is ",len(sentences))
    # print("words length is",len(corresponding_words))
    data={'Sentences':sentences,'Corresponding_words':corresponding_words}
    df = pd.DataFrame(data)
    df.to_csv('ptb_500.csv')
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
    a_list = [a for a in list if len(a) <= max and len(a) > min]
    return a_list

def count_avg_len(list):
    all_sen = [len(a) for a in list]
    # print(all_sen)
    avg = sum(all_sen)/len(all_sen)
    return avg
# path = '/Users/dyt/Documents/WORK/Yin/multinli_1.0/multinli_1.0_dev_matched.txt'
# path = '/Users/dyt/Documents/WORK/Yin/ptbdataset/ptb.train.txt'
path = '/Users/dyt/Documents/WORK/Yin/ptbdataset/ptb.test.txt'

sentence_list = extract_data(path)
sorted_list = sorted(sentence_list, key = len)
print(len(sorted_list))
print(count_avg_len(sorted_list))

# list_50 = len_interval(sorted_list,0, 50)
# list_100 = len_interval(sorted_list,50, 100)
# list_150 = len_interval(sorted_list,100, 150) #1175
# list_200 = len_interval(sorted_list,150, 200)
# list_250 = len_interval(sorted_list,200, 250)
# list_300 = len_interval(sorted_list,250, 300) #30
# list_350 = len_interval(sorted_list,300, 350) #12
# list_400 = len_interval(sorted_list,350, 400)#2
# list_450 = len_interval(sorted_list,400, 450)#0
# list_500 = len_interval(sorted_list,450, 500) #2
# give_bags_words(list_500)

# df = give_bags_words(sentence_list)
# df = give_count_words(sentence_list)