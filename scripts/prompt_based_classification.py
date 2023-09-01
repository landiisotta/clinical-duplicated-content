from openprompt.data_utils import InputExample
from datasets import load_dataset, DatasetDict
from collections import namedtuple
from openprompt.plms import load_plm, _MODEL_CLASSES, ModelClass
from transformers import MegatronBertConfig, MegatronBertForMaskedLM, BertTokenizer
from openprompt.plms.mlm import MLMTokenizerWrapper
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import time
from scripts.metrics import TaskMetrics
from tqdm import tqdm
from torch.optim import AdamW
import itertools
import os
import torch
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda")

ChallengeLabels = namedtuple('ChallengeLabels', ['smoking', 'cohort'])

SMOKING_CLASSES = [  # Smoking challenge classes (N=5)
    'NON-SMOKER',
    'SMOKER',
    'PAST SMOKER',
    'UNKNOWN'
]

LABELS_SMOKING = {0: 'NON-SMOKER',
                  1: 'SMOKER',
                  2: 'PAST SMOKER',
                  3: 'UNKNOWN'}

_MODEL_CLASSES['GatorTron'] = ModelClass(**{
    'config': MegatronBertConfig,
    'tokenizer': BertTokenizer,
    'model': MegatronBertForMaskedLM,
    'wrapper': MLMTokenizerWrapper,
})


def run_train_eval(smoking: DatasetDict, model_name: str, params: dict, tb_writer, seed=42) -> tuple:
    # Model
    plm, tokenizer, model_config, WrapperClass = load_plm("GatorTron", model_name)
    promptTemplate = ManualTemplate(
        text='{"placeholder":"text_a"}. Smoking history: {"mask"} smoker',
        tokenizer=tokenizer,
    )
    promptVerbalizer = ManualVerbalizer(
        classes=SMOKING_CLASSES,
        label_words={
            "NON-SMOKER": ["non", "not", "nonsmoker"],
            "SMOKER": ["current", "smoker", "yes"],
            "PAST SMOKER": ["former", "quit", "past"],
            "UNKNOWN": ["na", ".", "unknown", "never"]
        },
        tokenizer=tokenizer,
    )
    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
        freeze_plm=False
    )

    # Training set
    chk_classes = {0: 0, 1: 0, 2: 0, 3: 0}
    train_dataset = []
    for el in smoking['train']:
        if chk_classes[el['label']] == 1000:
            continue
        else:
            chk_classes[el['label']] += 1
            train_dataset.append(
                InputExample(
                    label=el['label'],
                    guid=el['id'],
                    text_a=el['note'],
                ))

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    train_data_loader = PromptDataLoader(
        dataset=train_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=params['batch_size'],
        shuffle=True
    )
    plm_lr = params['learning_rate']
    trained_model = train(promptModel, train_data_loader, plm_lr, steps=params['steps'],
                          tb_writer=tb_writer)
    # Test set
    test_dataset = []
    for el in smoking['test']:
        test_dataset.append(
            InputExample(
                label=el['label'],
                guid=el['id'],
                text_a=el['note'],
            ))
    test_data_loader = PromptDataLoader(
        dataset=test_dataset,
        tokenizer=tokenizer,
        template=promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=16,
        shuffle=False
    )
    output = eval(trained_model, test_data_loader)
    return output, trained_model


def train(model, train_loader, lr, steps, tb_writer):
    model.to(DEVICE)

    no_decay = ['bias', 'LayerNorm.weight']
    plm_param = [
        {'params': [p for n, p in model.plm.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.plm.named_parameters()
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    optimizer_fun = AdamW(plm_param, lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    total_steps = steps
    epochs = round(total_steps / len(train_loader)) + 1
    count_steps = 0
    for _ in tqdm(range(epochs)):
        if total_steps == 0:
            break
        total_loss = 0
        for step, batch in tqdm(enumerate(train_loader)):
            total_steps -= 1
            batch.to(DEVICE)
            logits = model(batch)
            labels = batch.label
            loss = loss_fn(logits, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer_fun.step()
            optimizer_fun.zero_grad()
            tb_writer.add_scalar('step_loss', total_loss / (step + 1), count_steps)
            count_steps += 1
            if total_steps == 0:
                break
            # print(f'Step {step} -- Loss: {total_loss / (step + 1)}')
        # print(f'Epoch loss: {total_loss / len(train_loader)}')
    tb_writer.close()
    return model


def eval(model, test_loader):
    test_metrics = TaskMetrics(challenge='smoking')
    model.eval()
    for batch in tqdm(test_loader):
        batch.to(DEVICE)
        with torch.no_grad():
            logits = model(batch)
            # out = model.prompt_model(batch)
        preds = torch.argmax(logits, dim=-1)

        test_metrics.add_batch(batch['label'].cpu().tolist(), preds.cpu().tolist())
        # for idx, lab in enumerate(batch['label']):
        #     true = lab.cpu().numpy()
        #     pred = preds[idx].cpu().numpy()
        #     if true != pred:
        #         print(f'True {LABELS_SMOKING[int(true)]} -- Pred {LABELS_SMOKING[int(pred)]}')
        #         ordered = np.argsort(out.logits[batch.loss_ids == 1][idx].cpu().detach().numpy())[::-1]
        #         print(tokenizer.convert_ids_to_tokens(ordered[:10]))

    out_metrics = test_metrics.compute()
    return out_metrics


def param_tuning(folds, dataset, pretrained_model, seed):
    param_grid = {'steps': [10, 20, 50, 100],
                  'batch_size': [2, 4, 8, 16],
                  'learning_rate': [1e-5, 2e-5, 5e-5]}
    # param_grid = {'steps': [10],
    #               'batch_size': [2],
    #               'learning_rate': [1e-5, 2e-5]}
    keys, values = zip(*param_grid.items())
    param_dict = [dict(zip(keys, v)) for v in itertools.product(*values)]
    best_f1_score = 0.0
    best_param = {}
    total_config = len(param_dict)
    count_config = 0
    best_model = None
    best_metrics = {}
    for param in param_dict:
        count_config += 1
        start = time.process_time()
        print(f'Running configuration {param}')
        writer = SummaryWriter(
            f'runs/prompt_based_smoking/tensorboard-pt{dataset.split("_")[0]}/{pretrained_model.split("/")[-1]}'
            f'hyp{param["learning_rate"]}-{param["steps"]}-{param["batch_size"]}-{seed}')
        out_metrics, out_model = run_train_eval(folds, pretrained_model, param, tb_writer=writer, seed=seed)
        if out_metrics['f1_score']['f1_micro'] > best_f1_score:
            best_f1_score = out_metrics['f1_score']['f1_micro']
            best_param = param
            best_model = out_model
            best_metrics = out_metrics
        print(out_metrics)
        end = round(time.process_time() - start, 2)
        print(f'Run {count_config}/{total_config} config in {end}s')
        print('\n')
    print(f'Best configuration: {best_param} -- F1 score: {best_f1_score}')
    return best_param, best_metrics, best_model


if __name__ == '__main__':
    # seeds = [42]
    seeds = [54]
    # seeds = [1234]
    # seeds = [0]
    # seeds = [8]
    # seeds = [1, 2, 3, 4, 5]
    # seeds = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    _PT_MODELS = []
    _DT_NAMES = []
    for pt_model, dt_name in itertools.product(_PT_MODELS, _DT_NAMES):
        smoking = load_dataset('clinical_notes/n2c2/challenge_datasets',
                               name=dt_name,
                               cache_dir='.cache/huggingface/datasets/')
        hp_folds = smoking['train'].train_test_split(test_size=0.25, seed=42)
        _, tokenizer, _, WrapperClass = load_plm("GatorTron", pt_model)
        promptTemplate = ManualTemplate(
            text='{"placeholder":"text_a"}. Smoking history: {"mask"} smoker',
            tokenizer=tokenizer,
        )
        testset = []
        for el in smoking['test']:
            testset.append(
                InputExample(
                    label=el['label'],
                    guid=el['id'],
                    text_a=el['note'],
                ))
        test_dl = PromptDataLoader(
            dataset=testset,
            tokenizer=tokenizer,
            template=promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            batch_size=16,
            shuffle=False
        )
        if pt_model == 'GatorTron':
            model_file_name = pt_model
        else:
            model_file_name = pt_model.split('dedup')[1].split('10000')[0]
        for seed in seeds:
            print(f'Training model {model_file_name} on {dt_name.split("-")[0]} with seed {seed}')
            out_param, out_metrics, out_model = param_tuning(hp_folds, dt_name, pt_model, seed)
            test_metrics = eval(out_model, test_dl)
            fixed_info = f'{model_file_name},{dt_name.split("_")[0]},{str(out_param["learning_rate"])},' \
                         f'{str(out_param["steps"])},{str(out_param["batch_size"])},{str(seed)},dev'
            if os.path.isfile('runs/prompt_based_smoking/prompt-based-smoking-results.csv'):
                f = open('runs/prompt_based_smoking/prompt-based-smoking-results.csv', 'a')
            else:
                f = open('runs/prompt_based_smoking/prompt-based-smoking-results.csv', 'w')
                header = 'model,dataset,learning_rate,steps,batch_size,seed,fold'
                for k, d in out_metrics.items():
                    for kk in d.keys():
                        header += f',{kk}'
                f.write(header + '\n')
            for k, d in out_metrics.items():
                for score in d.values():
                    fixed_info += f',{str(score)}'
            f.write(fixed_info + '\n')
            fixed_info_test = f'{model_file_name},{dt_name.split("_")[0]},,,,{str(seed)},test'
            for k, d in test_metrics.items():
                for score in d.values():
                    fixed_info_test += f',{str(score)}'
            f.write(fixed_info_test + '\n')
            f.close()
