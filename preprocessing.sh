# Dump clinical notes to csv file only retaining and uniforming relevant information
python3 scripts/dump_notes.py \
    --data_dir data \
    --file_name toy_dataset_2.csv \
    --save_dir data \
    --subj_id pat_id \
    --note_id note_id \
    --type note_type \
    --datetime note_datetime \
    --note_text text \
    --output_name DATASET2 \
    --only_train
# ALTERNATIVES
## Dump clinical notes train/validation to csv file
#python3 scripts/dump_notes.py \
#    --data_dir data \
#    --file_name toy_dataset_1.csv \
#    --save_dir data \
#    --subj_id pat_id \
#    --note_id note_id \
#    --type note_type \
#    --datetime note_datetime \
#    --note_text text \
#    --output_name DATASET1 \
#    --only_dev \
#    --split_ratio 0.3
## Dump clinical notes train/validation/test to csv file with same ratio
#python3 scripts/dump_notes.py \
#    --data_dir data \
#    --file_name toy_dataset_1.csv \
#    --save_dir data \
#    --subj_id pat_id \
#    --note_id note_id \
#    --type note_type \
#    --datetime note_datetime \
#    --note_text text \
#    --output_name DATASET1 \
#    --split_ratio 0.3

# Tokenize train notes
python3 scripts/note_tokenization.py \
  --save_dir data \
  --corpus_name DATASET2 \
  --split train \
  --tokenize
#ALTERNATIVE
## Tokenize alla available splits
#python3 scripts/note_tokenization.py \
#  --save_dir data \
#  --corpus_name DATASET1 \
#  --all_splits \
#  --tokenize

