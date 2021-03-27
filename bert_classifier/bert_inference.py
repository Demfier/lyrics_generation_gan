"""
This script runs BERT in inference mode for lyrics classification
"""
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification


BERT_TOKENIZER = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        cache_dir='bert-base-uncased-tokenizer-cache/'
)
BERT_CHECKPOINT = 'bert_models/checkpoint-1500'
BERT_MODEL = BertForSequenceClassification.from_pretrained(BERT_CHECKPOINT)
BERT_MODEL.eval()


def fix_apos(line):
    # handle cases like " 's" " 'll" " 're" " n't" etc
    apostrophe = r"\s'([a-z]+)"
    special = r"\s(n't)"
    return re.sub(special, r"n't", re.sub(apostrophe, r"'\1", line))


def text_processor(lines):
    """constructs sentences from the tokenized outputs"""
    return [fix_apos(l.strip()) for l in lines]


def score_text(text_batch, tokenizer, model):
    """
    text_batch is a list of sentences
    tokenizer is bert's tokenizer
    model is the bert model
    """

    text_batch = text_processor(text_batch)

    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        softmax_scores = torch.softmax(outputs.logits, dim=1)
        # extract the first dimension scores or the probability that a given
        # sentence is 'good' (has label 1)
        print(type(softmax_scores[:,1].cpu().numpy()[0].item()))
        return list(zip(text_batch, softmax_scores[:,1].cpu().numpy()))


if __name__ == '__main__':
    print(score_text(["a constant with a land",
                      "a old man 'll have n't four if you again"],
                      BERT_TOKENIZER,
                      BERT_MODEL))
