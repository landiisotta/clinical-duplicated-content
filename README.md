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

### Clinical not relevant duplicates classification
The aim of this step is to fine-tune a BERT-like model to classify duplicated sentences as clinically relevant or not 
relevant. Implementation includes: (1) sentences annotation; (2) task-adaptation of the (clinical) LM; 
(3) fine-tuning.

1. `create_task_dataset.py` enables the creation of files with sentences duplicated between notes 
(their counts are also reported). Sentences are split into train/dev/test with 60/20/20 ratios. 
From each dataset N sentences are sampled for annotation as relevant/not relevant. 
It is possible to specify a regex to guide the selection for possible not relevant duplicates 
(default is `[Aa]gree|[Pp]lease (return|call)`). Run as:

    ```
    python3 scripts/create_task_dataset.py --dataset_path data/DATASET1.train \
        --dupcontent_path data/DATASET1.train.remove.byterange.bysen \
        --save_dir data \
        --n_annotate 10 \
        --nr_regex '[Rr]eferences|[Ee]xternal [Ll]inks'
    ```
   
   OR
   
   ```
    python3 scripts/create_task_dataset.py --dataset_path data/DATASET2.train \
        --dupcontent_path data/DATASET2.train.remove.byterange.bysen \
        --save_dir data \
        --no_split
    ```

For the toy dataset we assume not relevant information to be external links and references.

    **Before moving to the task adaptation step, dumped sentences need to be manually annotated. To do that,
     only keep sentences that are relevant or not relevant, respectively, in the corresponding files. Add a second column
     with the counts of how many times those sentences are found in the corpus.
     Then, rename files with (sen,count) as:
        `data/train|dev|test.relevant|not-relevant-ANNOTATED`.**

2. The task-adaptive pretraining module enables the adaptation of a language model to the task at hand by 
further pretraining it on unlabeled examples. Run as:

```
python3 scripts/task_adaptive_pretraining.py \
  --dataset_name DATASET1 \
  --model GatorTron \
  --model_name GatorTron
```

Task-adapted models are saved in `runs/ta_pretrainingModelName`.

3. To fine-tune the task-adapted model run:

```
python3 scripts/dup_content_finetuning.py \
    --dataset_name DATASET1 \
    --tokenizer GatorTron \
    --model_name GatorTron
```

Change `params` dictionary to enable hyperparameter tuning.

4. With `bash not-relevant_content_extraction.sh` we run the following modules sequentially.

- The module `dup_content_predict.py` allows to soft label as either clinically relevant or not relevant the between-note
  duplicated sentences. The flag `--no_annotated` indicates that no sentences were manually annotated and hence they 
  don't need to be subtracted from the unlabeled sentence list.
    
  ```
   python3 scripts/dup_content_predict.py --dataset_name DATASET1 --tokenizer GatorTron --model_name GatorTron
  ```
  
  OR
  
  ```
  python3 scripts/dup_content_predict.py --dataset_name DATASET1 --tokenizer GatorTron --model_name GatorTron --no_annotated
  ```

- The module `extract_nrbyterange.py` allows the extraction of the byte offsets corresponding to the not relevant sentences. 
  After running:

  ```
  python3 scripts/extract_nrbyterange.py \
    --split train \
    --data_dir data \
    --dataset_name DATASET1 \
    --model_name GatorTron
  ```

5. To externally validate the fine-tuned model, we can use `extract_sen_manual_validation.py` on DATASET2 
(but also DATASET1 for manual validation) to extract N examples to check:

```
python3 scripts/extract_sen_manual_validation.py \
    --dataset_name DATASET1 \
    --model_name GatorTron \
    --data_dir data \
    --n_manual 10
```

### Dataset deduplication
The module `sample_dedup.py` randomly extracts N notes from each configuration obtained through the 
`deduplication_configurations.py` scripts in the corresponding folder. Multiple datasets are combined when deduplication
configurations are created. Sampling happens equally in all datasets combined (N/n each, where n is the number of
combined datasets). This module enables the creation of a subset of notes that can be used for faster domain adaptation.
Run as:
```
SAMPLE_SIZE=100
SEED=42
FOLD=train

python3 scripts/sample_dedup.py \
        --sample_size=$SAMPLE_SIZE \
        --seed=$SEED \
        --fold=$FOLD
```

### Domain adaptation
The domain adaptation phase is characterized by a (1) pretraining phase (`da_pretraining.py`) and an (2) evaluation 
phase (`da_eval.py`). 

- During the pretraining phase, the user provides the training configuration, the sample size, and 
  the seed used to generate the sampling, together with the model's name and hyperparameters. The selected model is then 
  further fine-tuned to adapt to different deduplicated corpora. PPL is evaluated on the training set as a sanity check.
  Best model is saved in `runs` folder.
  
  Run as:
  
  ```
  SAMPLE_SIZE=100
  SEED=42
  DEDUP_CONFIG=NONE
  MODEL=GatorTron
  STEPS=2
  MAX_SEQ_LEN=128
  BATCH_SIZE=8
  LEARNING_RATE=1e-5
  python3 scripts/da_pretraining.py \
    --sample_size=$SAMPLE_SIZE \
    --dedup_config=$DEDUP_CONFIG \
    --seed=$SEED \
    --model=$MODEL \
    --n_steps=$STEPS \
    --max_seq_length=$MAX_SEQ_LEN \
    --batch_size=$BATCH_SIZE \
    --lr=$LEARNING_RATE
  ```

- During the evaluation phase, the further pretrained models are evaluated on all deduplicated test set configurations 
  possibly using multiple seeds. PPL is saved in `experiments.txt`.
  
  Run as:
  
  ```
  SAMPLE_SIZE=100
  SEED_TRAIN=42
  SEED_TEST=42
  DEDUP_CONFIG_TRAIN=NONE
  DEDUP_CONFIG_TEST=WNBN
  MODEL=GatorTron
  MAX_SEQ_LEN=128
  BATCH_SIZE=8
  python3 scripts/da_eval.py \
    --sample_size=$SAMPLE_SIZE \
    --seed_train=$SEED_TRAIN \
    --seed_test=$SEED_TEST \
    --dedup_config_train=$DEDUP_CONFIG_TRAIN \
    --dedup_config_test=$DEDUP_CONFIG_TEST \
    --model_name=$MODEL \
    --max_seq_length=$MAX_SEQ_LEN \
    --batch_size=$BATCH_SIZE
  ```
   



---

[^1]
Wikipedia data was downloaded as

```
from datasets import load_dataset
wiki = load_dataset("wikipedia", "20220301.en")
```

We selected 1,000 wikipedia documents for dataset 1 and 500 for dataset 2.

   


  
