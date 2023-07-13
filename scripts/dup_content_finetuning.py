from datasets import DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, \
    DataCollatorWithPadding, EarlyStoppingCallback
from pathlib import Path
import numpy as np
import evaluate
import torch
from pynvml import *
from sklearn.model_selection import ParameterGrid
import random
import argparse
import sys
import shutil


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    dcmetrics.add_batch(predictions=predictions, references=labels)
    return dcmetrics.compute()


parser = argparse.ArgumentParser(description='Duplicated content classification task')
parser.add_argument('--tokenizer', help='Path to the tokenizer')
parser.add_argument('--model_name', help='Name of the task-adapted model', default='')
config = parser.parse_args(sys.argv[1:])

# Create task Dataset from annotated samples
label_dt = {'train': {},
            'dev': {},
            'test': {}}
for p in Path('data').glob('**/*ANNOTATED'):
    print(p)
    fold = str(p).split('/')[-1].split('.')[0]
    label = str(p).split('/')[-1].split('.')[-1].split('-')[0]
    with open(p, 'r') as f:
        for line in f:
            ll = line.split(',')
            label_dt[fold].setdefault('text', list()).append(ll[0])
            if 'not' in label:
                label_dt[fold].setdefault('label', list()).append(1)
            else:
                label_dt[fold].setdefault('label', list()).append(0)

for f in label_dt.keys():
    print(f'{f.upper()} label counts:')
    lab_c = np.unique(label_dt[f]['label'], return_counts=True)
    print(f'Relevant: {lab_c[1][0]}')
    print(f'Not-relevant: {lab_c[1][1]}')

label_dt = DatasetDict({k: Dataset.from_dict(label_dt[k]) for k in label_dt.keys()})
print('Fold counts:')
print(label_dt.flatten())

tokenizer = AutoTokenizer.from_pretrained(
    config.tokenizer)
tokenizer.add_tokens(['[DATE]', '[TIME]'], special_tokens=True)
tkn_dt = label_dt.map(tokenize_function, batched=True, num_proc=4)
tkn_dt = tkn_dt.remove_columns(['text'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
model = AutoModelForSequenceClassification.from_pretrained(
    f'runs/ta_pretraining{config.model_name}', num_labels=2)
if torch.cuda.is_available():
    model.to('cuda')
    print_gpu_utilization()

model.resize_token_embeddings(len(tokenizer))

# BEST HYPERPAR CONFIG
params = {
    'batch_size': [8],
    'epochs': [5],
    'learning_rate': [2e-5],
    'weight_decay': [0],
    'warmup_ratio': [0.01]
}
# params = {
#     'batch_size': [8, 16, 32],
#     'epochs': [5, 10, 20],
#     'learning_rate': [5e-5, 1e-5, 5e-6],
#     'weight_decay': [0, 0.1, 0.01],
#     'warmup_ratio': [0.01, 0.1, 0.2]
# }

if not os.path.isdir(f'runs/ta_finetuning{config.model_name}'):
    os.makedirs(f'runs/ta_finetuning{config.model_name}')

metrics_file = f'runs/ta_finetuning{config.model_name}/gs_finetuning_dev.metrics'
if os.path.isfile(metrics_file):
    skip = True
else:
    skip = False
f = open(metrics_file, 'w')
if not skip:
    f.write('batch_size,epochs,learning_rate,loss,f1,precision,recall\n')

best_model = []
best_precision = 0.0
tmp_trainer, tmp_comb = None, None
for comb in list(ParameterGrid(params)):
    print(f"Parameters: {comb}")
    training_args = TrainingArguments(
        output_dir=f'runs/ta_finetuning{config.model_name}',
        evaluation_strategy='epoch',
        eval_steps=1,
        # log_level='info',
        # logging_dir='/sc/arion/projects/mscic1/duplicated_content/runs/ta_finetuning',
        # logging_steps=1,
        # logging_strategy='epoch',
        # logging_first_step=True,
        weight_decay=comb['weight_decay'],
        warmup_ratio=comb['warmup_ratio'],
        num_train_epochs=comb['epochs'],
        learning_rate=comb['learning_rate'],
        per_device_train_batch_size=comb['batch_size'],
        per_device_eval_batch_size=comb['batch_size'],
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_precision',
        seed=42)
    dcmetrics = evaluate.load('dcmetrics')

    trainer = Trainer(model=model,
                      args=training_args,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
                      train_dataset=tkn_dt['train'],
                      eval_dataset=tkn_dt['dev'],
                      compute_metrics=compute_metrics,
                      data_collator=data_collator)
    results = trainer.train()
    results_eval = trainer.evaluate()
    print_summary(results)
    v = [comb['batch_size'], comb['epochs'], comb['learning_rate'], results.metrics['train_loss'],
         results_eval['eval_f1'], results_eval['eval_precision'], results_eval['eval_recall']]
    f.write(','.join([str(el) for el in v]) + '\n')

    if results_eval['eval_precision'] > best_precision:
        best_precision = results_eval['eval_precision']
        tmp_trainer = trainer
        tmp_comb = comb
    print('-' * 100)
    print('\n\n')

if tmp_trainer is not None:
    best_trainer = tmp_trainer
    best_comb = tmp_comb
    print(f'Best parameters configuration: {best_comb}')
    dev_pred = best_trainer.predict(tkn_dt['dev'])
    pred = np.argmax(dev_pred.predictions, axis=-1)
    pred_score = np.max(torch.nn.functional.softmax(torch.tensor(dev_pred.predictions), dim=-1).numpy(), axis=-1)
    i = 0
    errors = {'FP': [], 'FN': []}
    for pred_lab, true_lab in zip(pred, dev_pred.label_ids):
        # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tkn_dt['dev']['input_ids'][i])))
        if pred_lab != true_lab:
            if pred_lab == 1:
                errors['FP'].append((
                    tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tkn_dt['dev']['input_ids'][i])),
                    pred_score[i]))
                # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tkn_dt['dev'][i]['input_ids'])),
                #       'NOT-RELEVANT', 'TRUE_RELEVANT')
                # print('\n')
            else:
                errors['FN'].append((tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(tkn_dt['dev']['input_ids'][i])), pred_score[i]))
                # print(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tkn_dt['dev'][i]['input_ids'])),
                #       'RELEVANT', 'TRUE_NOT-RELEVANT')
                # print('\n')
        i += 1
    with open(f'runs/ta_finetuning{config.model_name}/error_analysis.csv',
              'w') as f:
        for k, vect in errors.items():
            if k == 'FP':
                for sen in vect:
                    f.write(sen[0] + ',' + 'PRED_NOT-RELEVANT' + ',' + 'TRUE_RELEVANT' + ',' + str(sen[1]) + '\n')
                f.write('\n')
            else:
                for sen in vect:
                    f.write(sen[0] + ',' + 'PRED_RELEVANT' + ',' + 'TRUE_NOT-RELEVANT' + ',' + str(sen[1]) + '\n')
    test_pred = best_trainer.predict(tkn_dt['test'])
    print(test_pred.metrics)

    model_dir = f'runs/ta_finetuning{config.model_name}'
    for d in os.listdir(model_dir):
        if 'checkpoint' in d:
            shutil.rmtree(os.path.join(model_dir, d))
    best_trainer.save_model(
        output_dir=f'runs/ta_finetuning{config.model_name}')

    # try_best = AutoModelForSequenceClassification.from_pretrained(
    #     '/sc/arion/projects/mscic1/duplicated_content/runs/ta_finetuning')
    # try_args = training_args
    # try_args.per_device_eval_batch_size = best_comb['batch_size']
    # try_trainer = Trainer(model,
    #                       args=training_args,
    #                       # train_dataset=tkn_dt['train'],
    #                       # eval_dataset=tkn_dt['dev'],
    #                       compute_metrics=compute_metrics,
    #                       data_collator=data_collator
    #                       )
    # try_metrics = try_trainer.predict(tkn_dt['test']).metrics
    # for k, v in test_pred.metrics.items():
    #     print(k, v, try_metrics[k])
    #     assert v == try_metrics[k]
else:
    print("Precision is 0.0 change something in your model's configuration and retry.")
f.close()
