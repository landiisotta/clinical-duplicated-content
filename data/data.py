import csv
import os
from pathlib import Path
import datasets

"""
Input: clinical corpus in `.csv` format with header "subject_id,note_id,note_type,note_datetime,text". 
Output: DatasetDict object for available splits. Datasets object with  
        features "subject_id,note_id,note_type,note_datetime,text".
"""

_DESCRIPTION = """
Loading clinical notes from ./data folder. Modify BUILDER_CONFIGS to adapt to different corpora.
"""
_FOLDER = os.path.join(os.getcwd(), 'data')


class ClinicalNotes(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("0.0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="DATASET1",
                               version=VERSION,
                               description="First dataset."),
        datasets.BuilderConfig(name="DATASET2",
                               version=VERSION,
                               description="Second dataset."),
    ]

    DEFAULT_CONFIG_NAME = "DATASET1"

    def _info(self):
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "note_id": datasets.Value("string"),
                "subject_id": datasets.Value("string"),
                "note_datetime": datasets.Value("string"),
                "note_type": datasets.Value("string")
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
        )

    def _split_generators(self, dl_manager):
        # If several configurations are possible (listed in BUILDER_CONFIGS),
        # the configuration selected by the user is in self.config.name
        data_dir = _FOLDER
        folds = Path(data_dir).glob(f'{self.config.name}.*.csv')
        return [
            datasets.SplitGenerator(
                name=getattr(datasets.Split, str(f).split('.')[-2].upper()),
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": str(f),
                },
            ) for f in folds]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            rd = csv.reader(f)
            next(rd)
            for key, data in enumerate(rd):
                yield key, {
                    "text": data[-1],
                    "note_datetime": data[3],
                    "note_type": data[2],
                    "note_id": data[1],
                    "subject_id": data[0]
                }
