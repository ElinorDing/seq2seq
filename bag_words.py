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
            # print(len(preprocessed[5]))
            # print(preprocessed[5])
            # sentence_list.append(preprocessed[6])
            # print(preprocessed[6])
    # print(sentence_list)
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
    df.to_csv('ptb_80.csv',index = False)
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
    all_sen = [re.split(' ',a) for a in list]
    # a_list = [a for a in list if len(a) <= max and len(a) > min]
    a_list  = []
    for a in all_sen:
        if len(a) <= max and len(a) > min:
            new_str = ' '.join(a)
            print(new_str)
            a_list.append(new_str)
            print('aaa: ', new_str)
            print('bbb: ', len(a))
    print(len(a_list))
    return a_list

def count_avg_len(list):
    for a in list:
        print(a)
        cl = re.split(' ',a)
    sentence_list = [len(re.split(' ',a)) for a in list]
    # all_sen = [len(a) for a in list]
    # print(all_sen)
    print('max',max(sentence_list))
    print('min',min(sentence_list))

    avg = sum(sentence_list)/len(sentence_list)
    return avg
# path = '/Users/dyt/Documents/WORK/Yin/multinli_1.0/multinli_1.0_dev_matched.txt'
# path = '/Users/dyt/Documents/WORK/Yin/ptbdataset/ptb.train.txt'
path = '/Users/dyt/Documents/WORK/Yin/ptbdataset/ptb.test.txt'

sorted_list = extract_data(path)
# df = give_bags_words(sentence_list)
# sorted_list = sorted(sentence_list, key = len)
# print(len(sorted_list))
# print(count_avg_len(sorted_list))


# list_60 = len_interval(sorted_list,0, 60) #700
# list_90 = len_interval(sorted_list,60, 90) #724
# list_120 = len_interval(sorted_list,90, 120) #806
# list_150 = len_interval(sorted_list,120, 150) #634
# list_500 = len_interval(sorted_list,150, 500) #

# list_13 = len_interval(sorted_list,0, 13) #692
# list_19 = len_interval(sorted_list,13, 19) # 805
# list_25 = len_interval(sorted_list,19, 25) #867
# list_31 = len_interval(sorted_list,25, 31) #667
list_80 = len_interval(sorted_list,31, 80) # 730

give_bags_words(list_80)

# df = give_bags_words(sentence_list)
# df = give_count_words(sentence_list)