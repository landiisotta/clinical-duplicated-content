import metrics as metrics
from transformers import AutoTokenizer
import torch
from torch.optim import AdamW
from transformers import MegatronBertForPreTraining, DataCollatorForLanguageModeling, BertForPreTraining
from transformers.optimization import get_polynomial_decay_schedule_with_warmup
from create_deduplicated_instances import TextDatasetForNextSentencePrediction
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import sys
import argparse

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CUDA_LAUNCH_BLOCKING = 1
GPUS = max(torch.cuda.device_count(), 1)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


def tokenize_function(examples):
    return tokenizer(examples['sentence'], max_length=128, truncation=True, padding=True)


def train(train_dataloader, vocab_size, model, optimizer, scheduler, steps):
    """Training pass"""
    model.train()

    loss_batches = 0
    train_metrics = metrics.LmMetrics(sample_size=len(train_dataloader.sampler))
    count_steps = 0
    batch_size = int(len(train_dataloader.sampler) / len(train_dataloader))
    for batch in train_dataloader:
        batch = batch.to(DEVICE)
        outputs = model(**batch)
        loss = outputs.loss
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        PROGRESS_BAR.update(1)

        loss_batches += loss.detach().sum() * batch['input_ids'].shape[0]
        train_metrics.compute_batch_metrics(mlm_logits=outputs.prediction_logits.detach(),
                                            nsp_logits=outputs.seq_relationship_logits.detach(),
                                            mlm_labels=batch['labels'].detach(),
                                            nsp_labels=batch['next_sentence_label'].detach(),
                                            vocab_size=vocab_size,
                                            dev=False)
        train_metrics.add_batch()
        steps -= 1
        count_steps += 1

        # Print loss every 10 steps
        if count_steps % 10 == 0:
            print(f"Epoch: {epoch}/Steps: {(count_steps)} -- Train loss: \
                  {loss_batches / (count_steps * batch_size)}")

        if steps == 0:
            return train_metrics.compute(), loss_batches / ((count_steps * batch['input_ids'].shape[0]) * GPUS), steps

    return train_metrics.compute(), loss_batches / (len(train_dataloader.sampler) * GPUS), steps


def test(test_dataloader,
         model,
         vocab_size):
    """Evaluation step"""
    model.eval()

    loss = 0
    evaluate_metrics = metrics.LmMetrics(sample_size=len(test_dataloader.sampler))
    for batch in tqdm(test_dataloader, total=len(test_dataloader), desc='Model evaluation'):
        # New labels for linear model PPL estimate (only based on last masked token)
        for idx, el in enumerate(batch['input_ids']):
            max_len = batch['input_ids'].shape[1]
            for i in range(max_len - 1, -1, -1):
                if el[i] == 102:
                    if el[i - 1] == 103:
                        break
                    else:
                        batch['labels'][idx][i - 1] = el[i - 1]
                        batch['input_ids'][idx][i - 1] = 103
                        break
        batch = batch.to(DEVICE)
        lm_labels = torch.tensor(create_lm_labels(batch['labels']))
        lm_labels = lm_labels.to(DEVICE)
        with torch.no_grad():
            outputs = model(**batch)

        evaluate_metrics.compute_batch_metrics(mlm_logits=outputs.prediction_logits,
                                               nsp_logits=outputs.seq_relationship_logits,
                                               mlm_labels=batch['labels'],
                                               lm_labels=lm_labels,
                                               nsp_labels=batch['next_sentence_label'],
                                               vocab_size=vocab_size,
                                               dev=True)

        evaluate_metrics.add_batch()
        loss += outputs.loss.sum().item() * batch['input_ids'].shape[0]
    return evaluate_metrics.compute(), loss / (len(test_dataloader.sampler) * GPUS)


def create_lm_labels(batch):
    """Create labels for language model, i.e., predict only last token before EOS [SEP]"""
    batch_size, seq_length = batch.shape
    lm_labels = []
    for idx in range(batch_size):
        lm_labels.append([-100] * seq_length)
        for padix in range(seq_length - 1, -1, -1):
            if batch[idx][padix] != -100:
                lm_labels[-1][padix] = batch[idx][padix].item()
                break
    return lm_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=str)
    parser.add_argument('--dedup_config', type=str)
    parser.add_argument('--seed', type=str, help='Seed used to generate the sample.')
    parser.add_argument('--model', type=str)
    parser.add_argument('--n_steps', type=int, default=5)
    parser.add_argument('--max_seq_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', '--lr', type=float, default=1e-5)
    config = parser.parse_args(sys.argv[1:])

    sample_size = config.sample_size
    dedup_config_train = config.dedup_config + sample_size + config.seed
    model_name = config.model
    n_steps = config.n_steps
    max_seq_length = config.max_seq_length
    batch_size = config.batch_size
    learning_rate = config.learning_rate

    # Import pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(['[DATE]', '[TIME]'], special_tokens=True)

    # Import pretrained model for further pretraining
    if model_name == 'GatorTron':
        model = MegatronBertForPreTraining.from_pretrained(model_name)
    else:
        model = BertForPreTraining.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.to(DEVICE)

    # Initialize optimizer and scheduler
    warmup_steps = int(n_steps / 100)
    total_steps = n_steps
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      betas=(0.9, 0.999),
                      eps=1e-8,
                      weight_decay=0.01)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer=optimizer,
                                                          num_warmup_steps=warmup_steps,
                                                          num_training_steps=total_steps,
                                                          lr_end=0.0,
                                                          power=1.0,
                                                          last_epoch=-1)

    print("Creating NSP instances for training set, loading cached version if available...")
    dt_train = TextDatasetForNextSentencePrediction(tokenizer,
                                                    os.path.join('data',
                                                                 f'{dedup_config_train}_sentences_train'),
                                                    block_size=max_seq_length, truncation=True)
    dt_train_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True,
                                                        mlm_probability=0.15)
    train_loader = DataLoader(dt_train,
                              collate_fn=dt_train_collator,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    if DEVICE == torch.device('cuda'):
        print(f"Using {GPUS} GPUs")
    else:
        print('GPUs not available, using CPU')
    print(f'Number of training instances {len(train_loader.sampler)} (batches = {len(train_loader)})')
    print(f"Number of training steps: {total_steps}")

    n_epochs = int(n_steps / len(train_loader))
    best_model_dir = f'./runs/da_pretraining{model_name}/dedup{dedup_config_train}tr'
    os.makedirs(best_model_dir, exist_ok=True)
    PROGRESS_BAR = tqdm()
    PROGRESS_BAR.total = total_steps

    loss_history = []
    start = time.process_time()
    for epoch in range(n_epochs + 1):
        # Train
        train_metrics, loss, n_steps = train(train_loader,
                                             len(tokenizer),
                                             model,
                                             optimizer,
                                             scheduler,
                                             n_steps)
        loss_history.append(loss)

        print('\n')
        print("*" * 100)
        print(f"Epoch: {epoch}/Steps: {(total_steps - n_steps)} -- Train metrics: {train_metrics}")
        print(f"Epoch: {epoch}/Steps: {(total_steps - n_steps)} -- Train loss: {loss}")
        print("*" * 100)
        print('\n')
    print(f'Domain adaptation (steps: {n_steps}) completed in {round(time.process_time() - start, 2)}')
    print(f'Evaluating model on training instances.')
    out_metrics, _ = test(train_loader, model, len(tokenizer))
    print(out_metrics)

    print('Saving model...')
    saving_start = time.process_time()
    if GPUS == 1:
        model.save_pretrained(best_model_dir)
    elif GPUS > 1:
        model.module.save_pretrained(best_model_dir)
    print(f'Model saved in {round(time.process_time() - saving_start, 2)}s')

    if os.path.isfile('experiments.txt'):
        f = open('experiments.txt', 'a')
    else:
        f = open('experiments.txt', 'w')
        f.write('dedup_config_train,dedup_config_test,model,epochs,steps,fold_eval,ppl,mlm_acc,nsp_acc\n')
    f.write(','.join([str(dedup_config_train),
                      '',
                      model_name,
                      str(n_epochs),
                      str(total_steps),
                      'train',
                      str(out_metrics['ppl']),
                      str(out_metrics['mlm_accuracy']),
                      str(out_metrics['nsp_accuracy'])]))
    f.write('\n')
    f.close()
