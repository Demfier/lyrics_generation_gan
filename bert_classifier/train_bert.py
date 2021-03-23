# prepare dataset

import os
import re

import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

RAW_DATA_PATH = 'data/raw/bert_classifier_data/'


def text_processor(file_path):
    """constructs sentences from the tokenized outputs"""

    # handle cases like " 's" " 'll" " 're" " n't" etc
    apostrophe = r'\s\'([a-z]+)|\s(n\'t)'
    with open(file_path) as f:
        return [re.sub(apostrophe, r"'\1", l.strip()) for l in f.readlines()]


def text_saver(text_lines, save_path):
    with open(save_path, 'w') as f:
        f.writelines(text_lines)


def read_lyrics_data(data_dir):
    lines = []
    labels = []

    # positve examples
    good_lines = text_processor(os.path.join(data_dir, 'good_lines.sorted'))
    lines += good_lines
    labels += [1] * len(good_lines)

    # the negative examples
    bad_lines = text_processor(os.path.join(data_dir, 'bad_lines.sorted'))
    lines += bad_lines
    labels += [0] * len(bad_lines)
    return lines, labels


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


train_texts, train_labels = read_lyrics_data(RAW_DATA_PATH)
train_texts, val_texts, train_labels, val_labels = \
    train_test_split(train_texts, train_labels, test_size=0.1)


tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    cache_dir='bert-base-uncased-tokenizer-cache/'
)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)


class LyricsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = LyricsDataset(train_encodings, train_labels)
val_dataset = LyricsDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    cache_dir='bert-base-uncased-model-cache/'
)

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)

trainer = Trainer(
    model=model,                  # the instantiated 🤗 Transformers model to be trained
    args=training_args,           # training arguments, defined above
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset     # evaluation dataset
)


trainer.train()

print(trainer.evaluate())
