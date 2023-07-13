# Build suffix array
python3 scripts/make_suffix_array.py data/DATASET1.train
# Find duplicates
cargo run self-similar --data-file data/DATASET1.train --length-threshold 100 --cache-dir tmp/cache --num-threads 8
# Collect duplicates
cargo run collect --data-file data/DATASET1.train --cache-dir tmp/cache \
 --length-threshold 100 > data/DATASET1.train.remove.byterange
# Within-note/between-note duplicated sentences collection
python3 scripts/duplicated_sentences.py --dataset data/DATASET1.train --remove_range data/DATASET1.train.remove.byterange
