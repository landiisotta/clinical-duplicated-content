
# Dump clinical notes to csv file only retaining and uniforming relevant information
python3 scripts/dump_notes.py \
    --data_dir data \
    --file_name toy_dataset_1.csv \
    --save_dir data \
    --subj_id pat_id \
    --note_id note_id \
    --type note_type \
    --datetime note_datetime \
    --note_text text \
    --output_name DATASET1 \
    --only_train
