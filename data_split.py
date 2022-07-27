import os
import pandas as pd
import random
from shutil import copy2


def data_set_split(src_data_folder1, target_data_folder1, train_scale=0.7, val_scale=0.1, test_scale=0.2):
    print("START THE SPLIT")
    class_name = os.listdir(src_data_folder1)[0]
    split_names = ['train', 'val', 'test']
    for split_name in split_names:
        split_path = os.path.join(target_data_folder1, split_name)
        if os.path.isdir(split_path):
            pass
        else:
            os.mkdir(split_path)

    current_class_data_path = os.path.join(src_data_folder1, class_name)
    print("current path: ",current_class_data_path)
    current_all_data = pd.read_csv(current_class_data_path)
    current_data_length = len(current_all_data)
    print(current_data_length)
    current_data_index_list = list(range(current_data_length))
    random.shuffle(current_data_index_list)
    train_folder = os.path.join(target_data_folder1, 'train')
    val_folder = os.path.join(target_data_folder1, 'val')
    test_folder = os.path.join(target_data_folder1, 'test')
    train_stop_flag = current_data_length * train_scale
    val_stop_flag = current_data_length * (train_scale + val_scale)
    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    df_train = pd.DataFrame(columns=['Sentences','Corresponding_words'])
    df_val = pd.DataFrame(columns=['Sentences','Corresponding_words'])
    df_test = pd.DataFrame(columns=['Sentences','Corresponding_words'])
    for i in current_data_index_list:
        print(i)
        print("type: ",type(current_all_data.iloc[i]))
        src_img_path = current_all_data.iloc[i]
        if current_idx <= train_stop_flag:
            df_train = df_train.append(src_img_path)
            # copy2(src_img_path, train_folder)
            train_num = train_num + 1
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            df_val = df_val.append(src_img_path)
            # copy2(src_img_path, val_folder)
            val_num = val_num + 1
        else:
            df_test = df_test.append(src_img_path)
            # copy2(src_img_path, test_folder)
            test_num = test_num + 1
        current_idx = current_idx + 1

    df_train.to_csv('training_clean.csv')
    copy2('training_clean.csv', train_folder)
    df_val.to_csv('val_clean.csv')
    copy2('val_clean.csv', val_folder)
    df_test.to_csv('test_clean.csv')
    copy2('test_clean.csv',test_folder)

    print("*********************************{}*************************************".format(class_name))
    print(
        "finished split {} according to {}：{}：{}，{} data in total".format(
            class_name, train_scale, val_scale, test_scale, current_data_length))
    print("train{}：{}".format(train_folder, train_num))
    print("dev{}：{}".format(val_folder, val_num))
    print("test{}：{}".format(test_folder, test_num))


if __name__ == '__main__':
    src_data_folder = r"/Users/dyt/workspace/seqToseq/raw_data"
    tar_data_folder = r"/Users/dyt/workspace/seqToseq/clean_data"
    data_set_split(src_data_folder, tar_data_folder)

