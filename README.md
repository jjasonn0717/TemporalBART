# TemporalBART

The source code for the paper "[Conditional Generation of Temporally-ordered Event Sequences](https://arxiv.org/abs/2012.15786)".

## Requirements

- torch 1.6.0
- allennlp 1.1.0
- transformers 3.0.2
- rouge-score
- graphviz

## Data

In this work, we specifically collect our training data from the [EventsNarratives](https://www.aclweb.org/anthology/P18-1050/) corpus, and evaluate our models in a zero-shot manner on CaTeRS and MCTaco dataset.

- The preprocessed training data can be downloaded from [here](https://drive.google.com/drive/folders/1Cyxi8meJ6TjeayVrnAeerN9FF7pnhyos?usp=sharing), where the 100K version is used for our best models.

- The preprocessed evaluation data can be found here: [CaTeRS](https://drive.google.com/file/d/1cEjt6Skb0Nnmy2a6XWEL6zVI57idT-Bj/view?usp=sharing), [MCTaco](https://drive.google.com/file/d/1z2V1_YUegKWVXjvaEyR0T-G_4AVAz6Ja/view?usp=sharing)

## Training

To train the TemporalBART models described in the paper, first change the data paths (`train_data_path, validation_data_path, test_data_path`) in the config file to where you store the downloaded data, then run the following commands:

- TemporalBART: `allennlp train ./configs/temporal_bart_config.json -s /PATH/TO/MODEL_CHKPT/ --include-package denoising_event_lm -f`

- TemporalBART-indexed: `allennlp train ./configs/temporal_bart_indexed_config.json -s /PATH/TO/MODEL_CHKPT/ --include-package denoising_event_lm -f`

## Evaluation

To evaluate the fine-tined models on temporal event ordering:

- CaTeRS:

`allennlp evaluate /PATH/TO/MODEL_CHKPT_TARGZ caters_entity_chains_seqdata_scrambled2.pkl --cuda-device 0 --include-package denoising_event_lm -o '{"validation_dataset_reader": {"chain_len_min": 2, "event_del_prob": 0.0, "event_del_all_prob": 0.0, "allow_empty_events": false}}'`

- MCTaco:
```
python3 denoising_event_lm/predictors/event_lm/test_demo_event_lm_mctaco_before_after.py \
    --archive-path /PATH/TO/MODEL_CHKPT_TARGZ \
    --predictor demo_denoising_event_lm_mctaco_before_after_predictor \
    --include-package denoising_event_lm \
    --cuda-device 0 \
    -o '{}' --input-path mctaco_event_ordering_before_after.json --beams 1 --feed-unseen > output.txt
```

To evaluate the fine-tuned models on ordering unseen events:

```
python3 denoising_event_lm/predictors/event_lm/test_demo_event_lm_orderextra.py \
    --archive-path /PATH/TO/MODEL_CHKPT_TARGZ \
    --predictor demo_denoising_event_lm_orderextra_predictor \
    --include-package denoising_event_lm \
    --cuda-device 0 \
    -o '{}' --input-path caters_entity_chains_seqdata_scrambled2.pkl --beams 2 --chain_len_min 2 > output.txt
```

## Model Checkpoints

The final models used in our paper can be downloaded with the following links:
- [TemporalBART](https://drive.google.com/file/d/1SdSrGhB4KMWIMzbD42GobKQmKPOIuRKL/view?usp=sharing)
- [TemporalBART-indexed](https://drive.google.com/file/d/1zYfYb-vLGBsXEu9rZ6KD3J5xC6mO3SVI/view?usp=sharing)
