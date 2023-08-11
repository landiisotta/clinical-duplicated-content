import argparse
from collections import Counter
from sklearn.model_selection import train_test_split
import re
import random as rnd
import os

parser = argparse.ArgumentParser(description='Create duplicated content splits and examples to annotate.')
parser.add_argument('--dataset_path', type=str)
parser.add_argument('--dupcontent_path', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--n_annotate', type=int, default=0)
parser.add_argument('--no_split', action='store_true')
parser.add_argument('--nr_regex', type=str, default='[Aa]gree|[Pp]lease (return|call)')

args = parser.parse_args()

data_path = args.dataset_path
dataset_name = data_path.split('/')[-1].split('.')[0]
dupcontent_path = args.dupcontent_path
save_dir = args.save_dir
n_annotate = args.n_annotate

rng = rnd.Random(42)


def read_note_and_content(dt_filepath,
                          dup_idx_filepath):
    dt = open(dt_filepath, 'rb').read()
    dup_content = []
    with open(dup_idx_filepath, 'r') as f:
        for line in f:
            if 'out' in line:
                break
        for line in f:
            idx = list(map(int, line.strip().split()))
            dup_content.append(dt[idx[0]:idx[1]])
    dup_count = Counter(dup_content)
    assert len(dup_count) == len(set(dup_content))
    return dt, dup_count


def create_train_validation_test_split(sentences):
    seed = 42
    train, test = train_test_split(sentences, train_size=0.60, random_state=seed)
    validation, test = train_test_split(test, train_size=0.50, random_state=seed)
    return train, validation, test


def dump_sentences(sentcounts, filename, filepath):
    with open(os.path.join(filepath, filename), 'w') as f:
        for el in sentcounts[:-1]:
            text = el[0].decode('utf8')
            c = str(el[1])
            f.write(','.join([text, c]))
            f.write('\n')
        f.write(','.join([sentcounts[-1][0].decode('utf8'), str(sentcounts[-1][1])]))


def extract_labels(dup_count, regex):
    # Extract possibly relevant and not relevant examples
    r_examples, nr_examples = [], []
    rng.shuffle(dup_count)
    for el in dup_count:
        if re.search(regex.encode('utf8'), el[0]):
            nr_examples.append(el)
        else:
            r_examples.append(el)
    return r_examples, nr_examples


def to_annotate(examples, n_examples):
    # Extract examples from list to annotate
    rng.shuffle(examples)
    return examples[:n_examples]


dataset, dupcontent_count = read_note_and_content(data_path, dupcontent_path)
if not args.no_split:
    dc_train, dc_validation, dc_test = create_train_validation_test_split(list(dupcontent_count.items()))
    dump_sentences(sorted(dc_validation, key=lambda x: x[1], reverse=True), f'{dataset_name}.validation.sen', save_dir)
    dump_sentences(sorted(dc_test, key=lambda x: x[1], reverse=True), f'{dataset_name}.test.sen', save_dir)
else:
    dc_train = list(dupcontent_count.items())
    print('Dumping all sentences to training set.')
dump_sentences(sorted(dc_train, key=lambda x: x[1], reverse=True), f'{dataset_name}.train.sen', save_dir)

if n_annotate > 0:
    # Train
    relevant, notrelevant = extract_labels(dc_train, args.nr_regex)
    dump_sentences(to_annotate(relevant, n_annotate), f'{dataset_name}.train.relevant', save_dir)
    dump_sentences(to_annotate(notrelevant, n_annotate), f'{dataset_name}.train.not-relevant', save_dir)
    if not args.no_split:
        # Dev
        relevant, notrelevant = extract_labels(dc_validation, args.nr_regex)
        dump_sentences(to_annotate(relevant, n_annotate), f'{dataset_name}.validation.relevant', save_dir)
        dump_sentences(to_annotate(notrelevant, n_annotate), f'{dataset_name}.validation.not-relevant', save_dir)
        # Test
        relevant, notrelevant = extract_labels(dc_test, args.nr_regex)
        dump_sentences(to_annotate(relevant, n_annotate), f'{dataset_name}.test.relevant', save_dir)
        dump_sentences(to_annotate(notrelevant, n_annotate), f'{dataset_name}.test.not-relevant', save_dir)
