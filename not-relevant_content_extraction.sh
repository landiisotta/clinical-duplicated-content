python3 scripts/dup_content_predict.py \
  --dataset_name DATASET1 \
  --tokenizer GatorTron \
  --model_name GatorTron

python3 scripts/extract_nrbyterange.py \
    --split train \
    --data_dir data \
    --dataset_name DATASET1 \
    --model_name GatorTron
