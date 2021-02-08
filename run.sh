#!/bin/bash
set -Exeuo pipefail

# For best performance, run 10 epochs. For seeing it run though, 5 epochs is OK.

# Runs both training jobs at the same time. If you run out of memory, run them one at a time.
allennlp train ./configs/temporal_bart_config.json -s ./temporal-bart-model --include-package denoising_event_lm -f -o '{"trainer": {"num_epochs": 5}}' &
allennlp train ./configs/temporal_bart_indexed_config.json -s ./temporal-bart-indexed-model --include-package denoising_event_lm -f -o '{"trainer": {"num_epochs": 5}}' &
wait -n
wait -n

# Evaluate both on CaTeRS
allennlp evaluate ./temporal-bart-model ./released_data_models/data/eval/caters_entity_chains_seqdata_scrambled2.pkl --cuda-device 0 --include-package denoising_event_lm -o '{"validation_dataset_reader": {"chain_len_min": 2, "event_del_prob": 0.0, "event_del_all_prob": 0.0, "allow_empty_events": false}}'
allennlp evaluate ./temporal-bart-indexed-model ./released_data_models/data/eval/caters_entity_chains_seqdata_scrambled2.pkl --cuda-device 0 --include-package denoising_event_lm -o '{"validation_dataset_reader": {"chain_len_min": 2, "event_del_prob": 0.0, "event_del_all_prob": 0.0, "allow_empty_events": false}}'

# Evaluate both on MCTaco
python3 denoising_event_lm/predictors/event_lm/test_demo_event_lm_mctaco_before_after.py \
    --archive-path ./temporal-bart-model \
    --predictor demo_denoising_event_lm_mctaco_before_after_predictor \
    --include-package denoising_event_lm \
    --cuda-device 0 \
    -o '{}' --input-path ./released_data_models/data/eval/mctaco_event_ordering_before_after.json --beams 1 --feed-unseen > mctaco-eval.txt
python3 denoising_event_lm/predictors/event_lm/test_demo_event_lm_mctaco_before_after.py \
    --archive-path ./temporal-bart-indexed-model \
    --predictor demo_denoising_event_lm_mctaco_before_after_predictor \
    --include-package denoising_event_lm \
    --cuda-device 0 \
    -o '{}' --input-path ./released_data_models/data/eval/mctaco_event_ordering_before_after.json --beams 1 --feed-unseen > mctaco-eval-indexed.txt

# Evaluate both on ordering unseen events
python3 denoising_event_lm/predictors/event_lm/test_demo_event_lm_orderextra.py \
    --archive-path ./temporal-bart-model/ \
    --predictor demo_denoising_event_lm_orderextra_predictor \
    --include-package denoising_event_lm \
    --cuda-device 0 \
    -o '{}' --input-path ./released_data_models/data/eval/caters_entity_chains_seqdata_scrambled2.pkl --beams 2 --chain_len_min 2 > unseen-eval.txt
python3 denoising_event_lm/predictors/event_lm/test_demo_event_lm_orderextra.py \
    --archive-path ./temporal-bart-indexed-model/ \
    --predictor demo_denoising_event_lm_orderextra_predictor \
    --include-package denoising_event_lm \
    --cuda-device 0 \
    -o '{}' --input-path ./released_data_models/data/eval/caters_entity_chains_seqdata_scrambled2.pkl --beams 2 --chain_len_min 2 > unseen-eval-indexed.txt
