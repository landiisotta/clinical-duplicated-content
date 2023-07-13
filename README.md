# Clinical Text Deduplication

Clinical notes are a unique source of information on patients' status and disease progression. Although large language 
models (LMs) have proven to learn medical knowledge, to some extent, they still require specialized domain
adaptation for improved downstream clinical tasks. 

Clinical notes are characterized by high levels of duplicated content and the extent to which such duplicates impact 
LMs and downstream tasks is still understudied.

This work includes:

1. The identification of duplicates defined as: _witin-note_; _between-note not relevant_; _between-note relevant_;
    - Clinical relevance is identified via a fine-tuned GatorTron model that classifies duplicates 
    as clinically relevant or not relevant.
    
2. Text deduplication: three possible configurations available. WN = clinical text w/o within-note duplicates;
WNNR = clinical text w/o within-note and between-note not relevant duplicates; WNBN = clinical text with all duplicates
dropped;

3. Investigation of the impact of deduplication practices on LMs via PPL;

4. Investigation of the impact of deduplication practices on downstream tasks via prompt-based learning.
 
By leveraging large real-world clinical corpora, we showed that such large datasets are characterized by high levels
 of duplication. We argue that clinical text deduplication, based on clinical relevance and duplicates distribution, 
 can help clinical LMs encode less redundant information in a more efficient manner and improve classification tasks
  via prompt-based learning.

**Remark**: the original work was performed on three clinical datasets: MIMIC-III v1.4, n2c2 challenge datasets, 
and ICU notes from the Mount Sinai Health System Data Warehouse (NYC). 
Access to such datasets is restricted, hence, we provide here a subset of the English Wikipedia dataset[^1] 
that was rearranged to resemble the structure of clinical corpora and can be used to test the code.

## Pipeline

### Clinical notes preprocessing
From root folder run
```
bash preprocessing.sh
```
to execute `dump_notes.py` (raw note file -> structured csv file with metadata) and `note_tokenization.py` 
(structured csv file -> dataset_name.split/dataset_name.split.size/dataset_mame.split.metadata). In `data` folder we
store the file with tokenized concatenated notes and their byte offsets, together with their metadata.

### Suffix array and duplication identification
The code to build the suffix array and find the duplicated text throughout the entire corpus was taken as is from 
[Lee et al., 2021](https://github.com/google-research/deduplicate-text-datasets/tree/master). To the pipeline, we added 
the module `duplicated_sentences.py` which enables the generation of the files `dataset_name.split.remove.byterange.wnr` 
(with the byte ranges of sentences duplicated within the same note) and `dataset_name.split.remove.byterange.bysen` 
(with all duplicated sentences). Duplicated text identified as in 
[Lee et al., 2021](https://aclanthology.org/2022.acl-long.577/) is dumped in `dataset_name.split.remove.byterange`, 
where byte ranges of duplicated text are saved. For each text chunk, we only retained sentences 
(starting with a capitalized character, ending with a period, and longer than 5 characters) and distinguished 
duplicated sentences within-note and between-note, respectively.

Execute the duplication identification steps via:

```
bash duplication_identification.sh
```
 

---

[^1]
Wikipedia data was downloaded as

```
from datasets import load_dataset
wiki = load_dataset("wikipedia", "20220301.en")
```

We selected 1,000 wikipedia documents for dataset 1 and 500 for dataset 2.

   


  
