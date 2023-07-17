import argparse
import os
from pathlib import Path
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Deduplicated datasets')
parser.add_argument('--split', type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--model_name', help='Fine-tuned model name',
                    default='')

args = parser.parse_args()
dataset_name = args.dataset_name

lab_filepath = os.path.join(args.data_dir, dataset_name + args.model_name + '.label')
nr_annt_filepath = Path(args.data_dir).glob(f'{dataset_name}*not-relevant-ANNOTATED')

byterange_filepath = os.path.join(args.data_dir, dataset_name + '.' + args.split + '.remove.byterange.bysen')

notes_filepath = os.path.join(args.data_dir, dataset_name + '.' + args.split)

output_filepath = os.path.join(args.data_dir,
                               dataset_name + '.' + args.split + f'.remove.byterange{args.model_name}.nr')

nr_sentences = set()
with open(lab_filepath, 'r') as f:
    for line in f:
        ll = line.strip().split(',')
        if ll[1] == 'NOT-RELEVANT':
            nr_sentences.add(ll[0])
for f in nr_annt_filepath:
    with open(f, 'r') as ff:
        for line in ff:
            nr_sentences.add(line.strip().split(',')[0])

dt = open(notes_filepath, 'rb').read()
nr_byteranges = []
with open(byterange_filepath, 'r') as f:
    for line in f:
        if 'out' in line:
            break
    for line in f:
        idx = list(map(str, line.strip().split()))
        if dt[int(idx[0]):int(idx[1])].decode('utf8') in nr_sentences:
            nr_byteranges.append(idx)

with open(output_filepath, 'w') as f:
    f.write('out\n')
    for idx in tqdm(nr_byteranges[:-1], desc='Saving not-relevant byte ranges', total=len(nr_byteranges)):
        f.write(' '.join(idx))
        f.write('\n')
    f.write(' '.join(nr_byteranges[-1]))
print(f"Saved {len(nr_byteranges)} byte ranges corresponding to not-relevant duplicates.")
