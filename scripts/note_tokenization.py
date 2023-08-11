"""
Module that enables clinical notes tokenization. The original dataset is tokenized and dumped concatenating
all documents as utf-8-encoded text in dataset_name.fold file.
File dataset_name.fold.size stores the byte offset where each individual example begins,
in sorted order. File dataset_name.fold.metadata stores patient ids, notes dates and times, and note types.
"""

from datasets import load_dataset
import os
import struct
import numpy as np
import datasets
import multiprocessing as mp
import argparse
from custom_tokenizer import tokenize


def sep():
    global UID
    UID += 1
    return pre_sep + struct.pack("<I", UID) + post_sep


def tok(x):
    if args.tokenize:
        out = tokenize(x)
    else:
        out = x
    return out


def concat_notes(ds, split):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fout = open(os.path.join(save_dir, dataset_name + "." + split), "wb")

    # Enabling batch mapping
    batch_size = 2 ** 16
    num_proc = mp.cpu_count() - 1
    if batch_size > len(ds):
        batch_size = len(ds)
        num_proc = 1
    ds = ds.map(lambda examples: {'text': tok(examples['text']),
                                  'subject_id': examples['subject_id'],
                                  'note_type': examples['note_type'],
                                  'note_id': examples['note_id'],
                                  'note_datetime': examples['note_datetime']},
                batch_size=batch_size,
                num_proc=num_proc)

    i = 0
    sizes = [0]
    metadata = []
    for el in ds:
        text = el['text'].encode('utf8')
        next_line = sep() + text
        fout.write(next_line)
        sizes.append(sizes[-1] + len(next_line))
        metadata.append([el[k] for k in el.keys() if k != 'text'])
        i += 1

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    open(os.path.join(save_dir, dataset_name + "." + split + ".size"), "wb").write(
        np.array(sizes, dtype=np.uint64).tobytes())
    # open(os.path.join(save_dir, dataset_name + "." + 'train' + ".metadata"), "wb").write(
    #     np.array([''] + [','.join(mdt) for mdt in metadata] + ['\n']).tobytes())
    with open(os.path.join(save_dir, dataset_name + "." + split + ".metadata"), "w") as f:
        f.write('note_id,subject_id,note_datetime,note_type\n')
        for mdt in metadata:
            f.write('' + ','.join([str(m) for m in mdt]) + '\n')
    print(f'Task ended for {split} split')


parser = argparse.ArgumentParser(description='Load a dataset.')
parser.add_argument('--save_dir', type=str)
parser.add_argument('--corpus_name', type=str)
parser.add_argument('--tokenize', action='store_true')
parser.add_argument('--split', type=str, default='train')
parser.add_argument('--all_splits', action='store_true')  #
parser.add_argument('--pre_sep', type=bytes, default=b"\xff\xff")
parser.add_argument('--post_sep', type=bytes, default=b"")

args = parser.parse_args()

save_dir = args.save_dir
dataset_name = args.corpus_name
split = args.split

pre_sep = args.pre_sep
post_sep = args.post_sep

# Text counter after pre_sep
UID = 0

ds = load_dataset(name=dataset_name,
                  path='data',
                  cache_dir='.cache/huggingface/datasets')
print(ds)
if not args.all_splits:
    ds = ds[split]
    assert isinstance(ds, datasets.Dataset)
    print(ds)
    concat_notes(ds, split)
else:
    for split in ['train', 'validation', 'test']:
        try:
            concat_notes(ds[split], split)
        except ValueError:
            print(f'Could not find split {split}')
            continue
