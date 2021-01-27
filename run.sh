#!/bin/bash
set -Exeuo pipefail

allennlp train ./configs/temporal_bart_config.json -s ./temporal-bart-model --include-package denoising_event_lm -f
allennlp train ./configs/temporal_bart_indexed_config.json -s ./temporal-bart-indexed-model --include-package denoising_event_lm -f
