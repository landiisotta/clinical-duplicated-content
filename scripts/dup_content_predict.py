from transformers import AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding, AutoTokenizer, \
    TrainingArguments
from datasets import DatasetDict, Dataset
from pathlib import Path
import numpy as np
import argparse
import torch


def read_file(file_path):
    sen_list = []
    with open(file_path, 'r') as f:
        for line in f:
            ll = line.split(',')
            if len(ll[0]) > 5:
                txt = ll[0]
                sen_list.append(txt)
            # s = re.search(br'\xff', ll[0])
            # if s:
            #     txt = ll[0][:s.span()[0]].decode('utf8')
            #     sen_list.append(txt)
            # else:
            #     try:
            #         txt = ll[0].decode('utf8')
            #         sen_list.append(txt)
            #     except UnicodeDecodeError:
            #         continue
    return sen_list


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128)


parser = argparse.ArgumentParser(description='Classify sentences into RELEVANT (0) and NOT-RELEVANT (1).')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--tokenizer', help='Path to the original tokenizer')
parser.add_argument('--model_name', help='Name of the finetuned model, e.g., ClinicalBERT or GatorTron', default='')
parser.add_argument('--no_annotated', action='store_true')

config = parser.parse_args()

convert_labels = {0: 'RELEVANT',
                  1: 'NOT-RELEVANT'}
print('*' * 100)
print(f'Processing duplicated sentences for dataset: {config.dataset_name.upper()}')
print('*' * 100)

# Store annotated sentences (no need to label them)
if not config.no_annotated:
    annt_files_gen = Path('data').glob(
        f'{config.dataset_name}*ANNOTATED')
    annt_sen_set = set()
    for p in annt_files_gen:
        print(p)
        annt_sen_set.update(read_file(p))
else:
    annt_files_gen, annt_sen_set = None, set()

# Collect sentences to be labeled
unlabeled_file_gen = Path('data').glob(
    f'{config.dataset_name}*.sen')
sen_out = []
for p in unlabeled_file_gen:
    print(p)
    sen_out.extend(list(set(read_file(p)).difference(annt_sen_set)))

dt = DatasetDict({'eval': Dataset.from_dict({'text': sen_out})})
print(dt)

# Load finetuned model for classification
tokenizer = AutoTokenizer.from_pretrained(
    config.tokenizer)
tokenizer.add_tokens(['[DATE]', '[TIME]'], special_tokens=True)
tkn_dt = dt.map(tokenize_function, batched=True, num_proc=4)
tkn_dt = tkn_dt.remove_columns(['text'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
    f'runs/ta_finetuning{config.model_name}/',
    num_labels=2)
if torch.cuda.is_available():
    model.to('cuda')

training_args = TrainingArguments(
    output_dir=f'runs/ta_finetuning{config.model_name}/',
    per_device_eval_batch_size=4)
trainer = Trainer(model,
                  args=training_args,
                  data_collator=data_collator)

pred_out = trainer.predict(tkn_dt['eval'])
pred_labels = np.argmax(pred_out.predictions, axis=1)

with open(
        f'data/{config.dataset_name}{config.model_name}.label',
        'w') as f:
    for sen, lab in zip(dt['eval']['text'][:-1], pred_labels[:-1]):
        f.write(','.join([sen, convert_labels[lab]]) + '\n')
    f.write(','.join([dt['eval']['text'][-1], convert_labels[pred_labels[-1]]]))
print('Predictions saved to file.')
