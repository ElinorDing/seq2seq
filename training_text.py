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
path = "/Users/dyt/workspace/seqToseq/training_data.csv"
df = pd.read_csv(path)
df = df.rename(columns = {"Sentences":"target_text","Corresponding_words":"source_text"})
df = df[['source_text', 'target_text']]
# df['source_text'] = "words to sentence: " + df['source_text']
# print(df.head(20))

# preprocessing

MODEL_NAME = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

inputs = preprocessing_function(df)

labels = inputs["labels"]
labels = torch.tensor(labels)
labels[labels == tokenizer.pad_token_id] = -100


# fine-tuning the model

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

batch_size = 16
output = model(
    input_ids = inputs["input_ids"],
    attention_mask = inputs["attention_mask"],
    labels = labels
)

# generating training/validation/test data
train_df, test_df = train_test_split(df, test_size=0.2)
print(train_df.shape, test_df.shape)
