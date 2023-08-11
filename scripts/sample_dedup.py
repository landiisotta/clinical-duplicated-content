"""
Script that randomly samples N notes from dataset configurations NONE, WN, WNNR, WNBN. Different seeds can be used for
different samplings.
"""

from datasets import load_dataset
import random
import argparse
import sys
from tqdm import tqdm


def sample_fold(sample_size, seed, fold):
    dt = {}
    for config in ['NONE', 'WN', 'WNNR', 'WNBN']:
        dt[config] = load_dataset('deduplication_configurations',
                                  name=config,
                                  cache_dir='.cache/huggingface/datasets')
    doc_ids = {}
    wnbn_doc_fold = list(set(dt['WNBN'][fold]['document']))
    for doc in wnbn_doc_fold:
        doc_ids.setdefault(doc.split('_')[0], list()).append(doc)

    rng = random.Random(seed)
    print(f"Sampling with seed {seed}")
    sample = []
    if len(doc_ids) == 1:
        sample += rng.sample(doc_ids[list(doc_ids.keys())[0]], sample_size)
    else:
        for dt_name in doc_ids:
            sample += rng.sample(doc_ids[dt_name], int(sample_size / 2))

    for config in ['NONE', 'WN', 'WNNR', 'WNBN']:
        print(f'Saving sample for {config} {fold} set.')
        PROGRESS_BAR = tqdm()
        PROGRESS_BAR.total = len(dt[config][fold])
        count = 0
        with open(f'data/{config}{sample_size}{seed}_sentences_{fold}', 'w') as f:
            id_ = 0
            while id_ < len(dt[config][fold]):
                doc = dt[config][fold][id_]['document']
                if doc in sample:
                    while dt[config][fold][id_]['document'] == doc:
                        f.write(dt[config][fold][id_]['sentence'] + '\n')
                        id_ += 1
                        PROGRESS_BAR.update(1)
                        if id_ == len(dt[config][fold]):
                            break
                    count += 1
                    if count == sample_size:
                        break
                    if id_ < len(dt[config][fold]):
                        f.write('\n')
                else:
                    PROGRESS_BAR.update(1)
                    id_ += 1
        print(f'Count: {count}\n\n')
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sample MIMIC/MSDW configurations by sample size.")
    parser.add_argument('--sample_size',
                        type=int,
                        dest='sample_size',
                        help="Size of the sample to extract (half from MIMIC, half from MSDW)")
    parser.add_argument('--seed',
                        type=int,
                        dest='seed',
                        help="Seed for random sampling")
    parser.add_argument('--fold',
                        type=str,
                        dest='fold',
                        help="Fold to extract the subsample from")
    args = parser.parse_args(sys.argv[1:])
    sample_fold(args.sample_size, args.seed, args.fold)
