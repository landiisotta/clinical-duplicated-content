"""Script to create a Dataset object with different deduplication configurations."""
import numpy as np
import os
import datasets
from collections import defaultdict
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Dataset description
_DESCRIPTION = """\
Available deduplication configurations (i.e., what sentences are removed from the original dataset are:
(1) NONE: raw data;
(2) WNBN: both within- and between-note duplicated content is removed;
(3) WN: only within-note redundancy is removed;
(4) WNNR: both within-note duplicated and not-relevant sentences are removed;
"""


def _build_lookup(rm_range_set, doc_bytes):
    num_notes = {}
    lookup_table = {}
    for i, _ in tqdm(enumerate(doc_bytes[:-1]), total=len(doc_bytes[:-1]), desc="Building lookup table"):
        num_notes[(doc_bytes[i], doc_bytes[i + 1])] = i
        for idx in range(doc_bytes[i], int(doc_bytes[i + 1] + 1)):
            if idx in rm_range_set and idx not in lookup_table:
                lookup_table[idx] = (doc_bytes[i], doc_bytes[i + 1])
    return num_notes, lookup_table


def _dedup_note(note, byte_spans):
    """
    Given a utf8-encoded note and a list of byte ranges, it returns
    the deduplicated version of the note.
    """
    note_rid = b''
    for i in range(len(byte_spans)):
        b = byte_spans[i][0]
        if i == 0:
            a = 0
        else:
            a = byte_spans[i - 1][1]
        if note[a:a + 3] == b' . ':
            note_rid += note[a + 3:b]
        else:
            note_rid += note[a:b]
    if note[byte_spans[-1][1]:byte_spans[-1][1] + 3] == b' . ':
        note_rid += note[byte_spans[-1][1] + 3:len(note)]
    else:
        note_rid += note[byte_spans[-1][1]:len(note)]
    note_rid = note_rid[6:].decode('utf8')
    if len(note_rid.strip()) > 0:
        return note_rid
    else:
        return None


def create_dataset_dict(file_names: list, config: str) -> list:
    remove = []
    if config == 'NONE':
        pass
    elif config == 'WNBN':
        for f in file_names:
            if 'bysen' in f or 'wnr' in f:
                fin = open(f)
                for line in fin:
                    if 'out' in line: break
                for line in fin:
                    remove.append(list(map(int, line.split())))
    elif config == 'WN':
        for f in file_names:
            if 'wnr' in f:
                fin = open(f)
                for line in fin:
                    if 'out' in line: break
                for line in fin:
                    remove.append(list(map(int, line.split())))
    elif config == 'WNNR':
        for f in file_names:
            if 'wnr' in f or 'byterange.nr' in f:
                fin = open(f)
                for line in fin:
                    if 'out' in line: break
                for line in fin:
                    remove.append(list(map(int, line.split())))
    return sorted(remove)


def sentences_to_remove(byterange_dir, config) -> dict:
    # Collect byte intervals for sentences to be removed
    file_names = [str(f) for f in Path(byterange_dir).glob(f'*.train.remove.byterange.*')]
    dataset_names = set([f.split('.')[0].split('/')[-1] for f in file_names])
    remove = {d: create_dataset_dict([f for f in file_names if d in f], config) for d in dataset_names}
    return remove


def deduplicate(suffixarray_dir, dataset_name, remove):
    sizes = np.frombuffer(open(os.path.join(suffixarray_dir, dataset_name + "." + "train.size"), "rb").read(),
                          dtype=np.uint64)
    dataset = open(os.path.join(suffixarray_dir, dataset_name + "." + 'train'), "rb").read()
    dedup_notes = {}
    remove_ex = defaultdict(list)
    remove_idx_set = set()

    if len(remove) > 0:
        # Note size?
        for el in remove:
            remove_idx_set.update(el)
        note_ids, lookup_table = _build_lookup(remove_idx_set, sizes)
        for ptr, _ in tqdm(enumerate(remove), total=len(remove)):
            if lookup_table[remove[ptr][0]] == lookup_table[remove[ptr][1]]:
                byte_start, byte_end = lookup_table[remove[ptr][0]]
                remove_ex.setdefault(lookup_table[remove[ptr][0]], list()).append(
                    (max(int(remove[ptr][0] - byte_start), 0),
                     min(int(remove[ptr][1] - byte_start),
                         byte_end - byte_start)))
            else:
                print("Found sentence spanning two notes! Please check your duplicates.")
                break
        del lookup_table
    else:
        note_ids = {}
        for i, _ in enumerate(sizes[:-1]):
            note_ids[(sizes[i], sizes[i + 1])] = i
    for k, idx in tqdm(note_ids.items(), desc=f"Extracting {dataset_name} notes", total=len(note_ids)):
        if k not in remove_ex:
            dedup_notes[idx] = dataset[k[0]:k[1]][6:].decode('utf8')
        else:
            note_rid = _dedup_note(dataset[k[0]:k[1]], remove_ex[k])
            if note_rid:
                dedup_notes[idx] = note_rid

    idx_train, idx_test = train_test_split(list(range(len(sizes))),
                                           test_size=0.40,
                                           random_state=42,
                                           shuffle=True)

    dedup_train = {k: dedup_notes[k].split(' . ') for k in idx_train if k in dedup_notes}
    dedup_test = {k: dedup_notes[k].split(' . ') for k in idx_test if k in dedup_notes}
    return dedup_train, dedup_test


class DeduplicationConfigurations(datasets.GeneratorBasedBuilder):
    """Each configuration corresponds to notes from all the available datasets
    w/o the corresponding duplicated sentences.
    NONE: no duplicated content is removed; WNBN: all duplicated content found is removed; WN: content repeated
    within-note is removed; WNNR: within-note and not-relevant duplicated content is removed."""

    VERSION = datasets.Version("0.0.1")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name='NONE',
                               version=VERSION,
                               description="No duplicated content is removed",
                               data_dir='data'),
        datasets.BuilderConfig(name='WNBN',
                               version=VERSION,
                               description="All duplicated content is removed",
                               data_dir='data'),
        datasets.BuilderConfig(name='WN',
                               version=VERSION,
                               description="Within-note duplicated content is removed",
                               data_dir='data'),
        datasets.BuilderConfig(name='WNNR',
                               version=VERSION,
                               description="Within-note and not-relevant duplicated content is removed",
                               data_dir='data'),
    ]

    DEFAULT_CONFIG_NAME = 'NONE'

    byterange_dir = 'data'

    def _info(self):
        features = datasets.Features(
            {
                "sentence": datasets.Value("string"),
                "document": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # Files are organized as document_id, sentence, e.g.,
        # MSDW_01,Odi et amo quare id faciam fortasse requiris
        # Splits only include training and test sets, this dataset configuration
        # will only be used for domain-adaptive pretraining of the ClinicalBERT model.
        remove_sentences = sentences_to_remove(byterange_dir=self.byterange_dir,
                                               config=self.config.name)
        dedup_train, dedup_test = {}, {}
        for dt_name, sen_span in remove_sentences.items():
            dedup_notes_tr, dedup_notes_ts = deduplicate(suffixarray_dir=self.config.data_dir,
                                                         dataset_name=dt_name,
                                                         remove=sen_span)

            dedup_train |= {dt_name + '_' + str(k): dedup_notes_tr[k] for k in dedup_notes_tr}
            dedup_test |= {dt_name + '_' + str(k): dedup_notes_ts[k] for k in dedup_notes_ts}

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dedup_notes": dedup_train,
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "dedup_notes": dedup_test,
                    "split": "test"
                },
            )
        ]

    def _generate_examples(
            self, dedup_notes, split  # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    ):
        """ Yields examples as (key, example) tuples. """
        # checkpoint = 'model/pretrained_bert_tf/biobert_pretrain_output_all_notes_150000'
        # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        # tokenizer.add_tokens(['[DATE]', '[TIME]'], special_tokens=True)
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        id_ = -1
        if not os.path.isfile(f'data/{self.config.name}_sentences_{split}'):
            f = open(f'data/{self.config.name}_sentences_{split}', 'w')
            for d, n in dedup_notes.items():
                if id_ > 0:
                    f.write('\n')
                for sen in n:
                    if len(sen) > 0:
                        f.write(sen + '\n')
                        id_ += 1
                        yield id_, {
                            "sentence": sen,
                            "document": d,
                        }
            f.close()
        else:
            for d, n in dedup_notes.items():
                for sen in n:
                    if len(sen) > 0:
                        id_ += 1
                        yield id_, {
                            "sentence": sen,
                            "document": d,
                        }
