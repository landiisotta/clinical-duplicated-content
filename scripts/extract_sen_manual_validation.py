import random
import argparse
import sys
import os

rng = random.Random(42)


def read_pred_file(file_path):
    sen_list, label_list = [], []
    with open(file_path, 'r') as f:
        for line in f:
            ll = line.split(',')
            if len(ll[0]) <= 5:
                break
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
            label_list.append(ll[1])
    return sen_list, label_list


def dump_sen(pred_sen_dict, n, dataset_name, model_name, data_path):
    for label in ['relevant', 'not-relevant']:
        output_name = os.path.join(data_path, f'{dataset_name}{model_name}.validate.label')
        with open(output_name, 'w') as f:
            try:
                sents = rng.sample(pred_sen_dict[label.upper()], n)
            except ValueError:
                sents = pred_sen_dict[label.upper()]
            for s in sents[:-1]:
                f.write(s)
                f.write('\n')
            f.write(sents[-1])


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name')
parser.add_argument('--model_name')
parser.add_argument('--data_dir')
parser.add_argument('--n_manual', type=int)

args = parser.parse_args(sys.argv[1:])

file = os.path.join(args.data_dir, f'{args.dataset_name}{args.model_name}.label')

print(f'Prediction for {args.dataset_name.upper()}:')
pred_dict = {'RELEVANT': [],
             'NOT-RELEVANT': []}
sen_list, label_list = read_pred_file(file)
for sen, lab in zip(sen_list, label_list):
    pred_dict[lab.strip()].append(sen)
print(
    f'RELEVANT (%): {len(pred_dict["RELEVANT"])} ({len(pred_dict["RELEVANT"]) / (len(pred_dict["RELEVANT"]) + len(pred_dict["NOT-RELEVANT"]))})')
print(
    f'NOT_RELEVANT (%): {len(pred_dict["NOT-RELEVANT"])} ({len(pred_dict["NOT-RELEVANT"]) / (len(pred_dict["RELEVANT"]) + len(pred_dict["NOT-RELEVANT"]))})')
print('\n')
dump_sen(pred_dict, args.n_manual, args.dataset_name, args.model_name, args.data_dir)
