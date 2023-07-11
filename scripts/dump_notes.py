"""
This module takes as input a .csv file with clinical notes and returns a standardized .csv file
retaining relevant columns: subject id, note id, date and time, note type, and text. If enabled, output can also
be split into train.dev/test folds.
"""
import random
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import date

rng = random.Random(42)

parser = argparse.ArgumentParser(description="Create train/dev/test splits.")
parser.add_argument('--data_dir', type=str, help='Folder where input is stored')
parser.add_argument('--file_name', type=str, help='Notes file name')
parser.add_argument('--output_name', type=str, help='Output file name')
parser.add_argument('--save_dir', type=str, help='Output directory')
parser.add_argument('--subj_id', type=str)
parser.add_argument('--note_id', type=str, default=None)
parser.add_argument('--datetime', type=str, default=None)
parser.add_argument('--type', type=str, default=None)
parser.add_argument('--note_text', type=str)
parser.add_argument('--only_dev', action='store_true')
parser.add_argument('--only_train', action='store_true')
parser.add_argument('--split_ratio', type=float, default=0.0, help='Ratio of dev set examples. If both only_dev and '
                                                                   'only_train are set to False, then the ratio refers '
                                                                   'to both dev and test sets.')

args = parser.parse_args()

data_dir = args.data_dir
file_name = args.file_name
subject_id = args.subj_id
note_text = args.note_text
note_datetime = args.datetime
note_type = args.type
save_dir = args.save_dir
split_ratio = args.split_ratio

print(f"Creating splits for {file_name.split('.')[0]} on {date.today()}.")
dt = pd.read_csv(os.path.join(data_dir, file_name), header=0, low_memory=False)
subj_id = dt[subject_id].unique()
splits = {}
# Case with no note id
if args.note_id is None:
    dt['NOTE_ID'] = [i + 1 for i in range(dt.shape[0])]
    note_id = "NOTE_ID"
else:
    note_id = args.note_id

if args.only_dev:
    print(f"Creating dev set ({split_ratio * 100}%) split from training notes.")
    if split_ratio <= 0.0:
        raise ValueError('Split ratio should be > 0.0')
    idx_train, idx_dev = train_test_split(subj_id, test_size=split_ratio, random_state=42)
    splits['test'] = None
elif args.only_train:
    idx_train = subj_id
    idx_dev = None
else:
    if split_ratio >= 0.5:
        raise ValueError('Split ratio refers to both dev and test sets. It should be < 0.5 to retain any example in '
                         'training set.')
    print(
        f"Creating train/dev/test ({100 - (split_ratio * 200)}/{split_ratio * 100}/{split_ratio * 100}%) "
        f"splits from notes.")
    if split_ratio <= 0.0:
        raise ValueError('Split ratio should be > 0.0')
    idx_train, idx_test = train_test_split(subj_id, test_size=split_ratio * 2, random_state=42)
    idx_dev, idx_test = train_test_split(idx_test, test_size=0.5, random_state=42)
    splits['test'] = idx_test

splits['train'] = idx_train
splits['dev'] = idx_dev

for s, idx in splits.items():
    if idx is not None:
        if args.note_id is not None:
            dt_split = dt.loc[dt[subject_id].isin(idx)][[subject_id, note_id, note_type,
                                                         note_datetime, note_text]].rename(
                columns={subject_id: "subject_id",
                         note_id: "note_id",
                         note_type: "note_type",
                         note_datetime: "note_datetime",
                         note_text: "text"})
        else:
            dt_split = dt.loc[dt[subject_id].isin(idx)][[subject_id, note_text]].rename(
                columns={subject_id: 'subject_id', note_text: "text"})
        print(f"Saving {s} set: N notes {dt_split.shape[0]} ({round((dt_split.shape[0] / dt.shape[0]) * 100, 2)}%) "
              f"-- N patients {dt_split.subject_id.nunique()} "
              f"({round((dt_split.subject_id.nunique() / len(subj_id)) * 100, 2)}%)")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        dt_split.to_csv(os.path.join(save_dir, f'{args.output_name}.{s}.csv'), index_label=None, index=False)
