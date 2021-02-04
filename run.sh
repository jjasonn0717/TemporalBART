#!/bin/bash
set -Exeuo pipefail

# For best performance, run 10 epochs. For seeing it run though, 5 epochs is OK.

# Runs both training jobs at the same time. If you run out of memory, run them one at a time.
allennlp train ./configs/temporal_bart_config.json -s ./temporal-bart-model --include-package denoising_event_lm -f -o '{"trainer": {"num_epochs": 5}}' &
allennlp train ./configs/temporal_bart_indexed_config.json -s ./temporal-bart-indexed-model --include-package denoising_event_lm -f -o '{"trainer": {"num_epochs": 5}}' &
wait -n
wait -n

