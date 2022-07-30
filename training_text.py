import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from datasets import load_from_disk

# test
max_source_length = 512
max_target_length = 128

def preprocess_function(examples):
    inputs = ["words to sentence: " + doc for doc in examples["source_text"]]
    model_inputs = tokenizer(inputs, padding="longest",max_length=max_source_length,
                             truncation=True,return_tensors='pt')

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["target_text"], padding="longest",
                           max_length=max_target_length, truncation=True,return_tensors='pt')

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# test


# gathering and clean dataset

# ready_dataset = load_from_disk("train_dataset")
ready_dataset = pd.read_csv("~/workspace/seqToseq/seq2seq/clean_data/train/training_clean.csv")
# preprocessing

MODEL_NAME = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

tokenized_datasets = ready_dataset.map(preprocess_function, batched=True)
print("The content of map is: ",tokenized_datasets)
# inputs = preprocessing_function(df)

labels = tokenized_datasets["labels"]
# labels = torch.tensor(labels)
labels[labels == tokenizer.pad_token_id] = -100


# fine-tuning the model

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

batch_size = 16
output = model(
    input_ids = tokenized_datasets["input_ids"],
    attention_mask = tokenized_datasets["attention_mask"],
    labels = labels
)