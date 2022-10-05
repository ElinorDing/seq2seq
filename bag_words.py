import pandas as pd
import re
import spacy
from collections import Counter
from collections import defaultdict

def extract_data(path):
    sentence_list = []
    # lines = []
    with open(path) as file_in:
        for line in file_in:
            # print(line)
            preprocessed = re.split(r'\s{3,}',line.replace('\t', '   ').replace('\n', ''))
            # lines.append(preprocessed)
            sentence_list.append(preprocessed[5])
            sentence_list.append(preprocessed[6])
    # print(len(sentence_list))
    return sentence_list

# def give_bags_words(sentences):
#     # sen_with_words = defaultdict(list)
#     nlp = spacy.load("en_core_web_sm")
#     count = 1
#     corresponding_words = []
#     for one in sentences:
#         doc = nlp(one)
#         corresponding_words.append([token.text for token in doc])
#         count += 1
#         print(count)
#     # print("sentences length is ",len(sentences))
#     # print("words length is",len(corresponding_words))
#     data={'Sentences':sentences,'Corresponding_words':corresponding_words}
#     df = pd.DataFrame(data)
#     # df.to_csv('training_data.csv')
#     return df

def give_count_words(sentences):
    # sen_with_words = defaultdict(list)
    nlp = spacy.load("en_core_web_sm")
    count = 1
    corresponding_words = []
    for one in sentences:
        doc = nlp(one)
        item_list = [token.text for token in doc]
        counts = Counter(item_list)
        counts_list = [(k, v) for k, v in counts.items()]
        count += 1
        print(count)
        print(counts_list)
        corresponding_words.append(counts_list)
    # print("sentences length is ",len(sentences))
    # print("words length is",len(corresponding_words))
    data={'Sentences':sentences,'Corresponding_words':corresponding_words}
    df = pd.DataFrame(data)
    df.to_csv('count_training_data.csv')
    return df

path = '/Users/dyt/Documents/WORK/Yin/multinli_1.0/multinli_1.0_dev_matched.txt'
sentence_list = extract_data(path)
# df = give_bags_words(sentence_list)
df = give_count_words(sentence_list)