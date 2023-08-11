from da_pretraining import test, create_lm_labels
import torch
from create_deduplicated_instances import TextDatasetForNextSentencePrediction
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, MegatronBertForPreTraining, BertForPreTraining
from torch.utils.data import DataLoader
import os
import sys
import argparse

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CUDA_LAUNCH_BLOCKING = 1
GPUS = max(torch.cuda.device_count(), 1)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--sample_size', type=str)
parser.add_argument('--seed_train', type=str, help='Seed used to generate the training sample.')
parser.add_argument('--seed_test', type=str, help='Seed used to generate the test sample.')
parser.add_argument('--dedup_config_train', type=str)
parser.add_argument('--dedup_config_test', type=str)
parser.add_argument('--model_name', type=str)
parser.add_argument('--max_seq_length', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
config = parser.parse_args(sys.argv[1:])

sample_size = config.sample_size
dedup_config_train = config.dedup_config_train + sample_size + config.seed_train
dedup_config_test = config.dedup_config_test + sample_size + config.seed_test
model_name = config.model_name
max_seq_length = config.max_seq_length
batch_size = config.batch_size

# Import pretrained tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(['[DATE]', '[TIME]'], special_tokens=True)

# Import further pretrained model
best_model_dir = f"./runs/da_pretraining{model_name}/dedup{dedup_config_train}tr"
if not os.path.isdir(best_model_dir):
    raise NotImplementedError('Before evaluation, run domain adaptation on training set via "da_pretraining.py".')
if model_name == 'GatorTron':
    model = MegatronBertForPreTraining.from_pretrained(best_model_dir, from_tf=False)
else:
    model = BertForPreTraining.from_pretrained(best_model_dir, from_tf=False)

model.to(DEVICE)

print("Creating NSP instances for test set, loading cached version if available...")
testset = TextDatasetForNextSentencePrediction(tokenizer,
                                               os.path.join('data', f'{dedup_config_test}_sentences_test'),
                                               block_size=max_seq_length, truncation=True)

test_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                mlm=True,
                                                mlm_probability=0.15)
test_loader = DataLoader(
    testset,
    batch_size=batch_size,
    collate_fn=test_collator,
    num_workers=0
)

if DEVICE == torch.device('cuda'):
    print(f"Using {GPUS} GPUs")
else:
    print('GPUs not available, using CPU')
print(f'Only evaluating model: instances (batches) {len(test_loader.sampler)} ({len(test_loader)})')
out_metrics, _ = test(test_loader, model, len(tokenizer))
print(f"Model trained on redundancy {dedup_config_train}, tested on redundancy {dedup_config_test}:")
print(out_metrics)

with open('experiments.txt', 'a') as f:
    f.write(','.join([str(dedup_config_train),
                      str(dedup_config_test),
                      model_name,
                      '',
                      '',
                      'test',
                      str(out_metrics['ppl']),
                      str(out_metrics['mlm_accuracy']),
                      str(out_metrics['nsp_accuracy'])]))

    f.write('\n')
