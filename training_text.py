import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# test
max_source_length = 1024
max_target_length = 128

def preprocessing_function(examples):
    inputs = ["words to sentence: " + doc for doc in examples["source_text"]]
    model_inputs = tokenizer(inputs, padding="longest",max_length=max_input_length,
                             truncation=True, return_tensors="pt")

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], padding="longest",
                           max_length=max_target_length, truncation=True, return_tensors="pt")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# test


# gathering and clean dataset

def merge_dataset(train, val, test):
    df_train = pd.read_csv(train)
    df_train = df_train.rename(columns = {"Sentences":"target_text","Corresponding_words":"source_text"})
    df_train = df_train[['source_text', 'target_text']]
    df_val = pd.read_csv(val)
    df_val = df_val.rename(columns = {"Sentences":"target_text","Corresponding_words":"source_text"})
    df_val = df_val[['source_text', 'target_text']]
    df_test = pd.read_csv(test)
    df_test = df_test.rename(columns = {"Sentences":"target_text","Corresponding_words":"source_text"})
    df_test = df_test[['source_text', 'target_text']]

    train_dataset = datasets.Dataset.from_dict(df_train)
    val_dataset = datasets.Dataset.from_dict(df_val)
    test_dataset = datasets.Dataset.from_dict(df_test)
    my_dataset_dict = datasets.DatasetDict({"train":train_dataset,"val":val_dataset,"test":test_dataset})
    print(my_dataset_dict)
    return my_dataset_dict

train = r"/Users/dyt/workspace/seqToseq/clean_data/train/training_clean.csv"
val = r"/Users/dyt/workspace/seqToseq/clean_data/val/val_clean.csv"
test = r"/Users/dyt/workspace/seqToseq/clean_data/test/test_clean.csv"
ready_dataset = merge_dataset(train,val,test)

# path = "/Users/dyt/workspace/seqToseq/training_data.csv"
# df = pd.read_csv(path)
# df = df.rename(columns = {"Sentences":"target_text","Corresponding_words":"source_text"})
# df = df[['source_text', 'target_text']]
# df['source_text'] = "words to sentence: " + df['source_text']
# print(df.head(20))

# preprocessing

MODEL_NAME = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

tokenized_datasets = ready_dataset.map(preprocess_function, batched=True)
# inputs = preprocessing_function(df)

labels = tokenized_datasets["labels"]
labels = torch.tensor(labels)
labels[labels == tokenizer.pad_token_id] = -100


# fine-tuning the model

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

batch_size = 16
output = model(
    input_ids = tokenized_datasets["input_ids"],
    attention_mask = tokenized_datasets["attention_mask"],
    labels = labels
)