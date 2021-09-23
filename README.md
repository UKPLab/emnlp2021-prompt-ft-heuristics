# Avoiding Inference Heuristics in Prompt-based Finetuning

## Quick links

* [Overview](#overview)
* [Requirements](#requirements)
* [Prepare the data](#prepare-the-data)
* [Run the model](#run-lm-bff)
  * [Quick start](#quick-start)

## Overview
The code includes implementation to reproduce results shown in our submitted paper. The experiments include:
1. Standard non-prompt finetuning on MNLI, QQP, and SNLI.
2. Few-shot finetuning using prompts on MNLI, QQP, and SNLI. Trained model on each dataset will be evaluated against
 HANS, PAWS, and Scramble Test, respectively.
3. Few-shot finetuning trained using regularized objectives such as L2 loss between the updated and the pretrained
 weights; and partial layers freezing.

This implementation is based on LM-BFF by [Gao et al (2020)](https://arxiv.org/pdf/2012.15723.pdf).

## Requirements

To run the code, please install all the dependencies with the following command:

```
pip install -r requirements.txt
```

## Prepare the data

The original datasets (MNLI, SNLI, QQP) can be downloaded here [here
](https://nlp.cs.princeton.edu/projects/lm-bff/datasets.tar). Please download it and extract the files to `./data/original`, or run the following commands:

```bash
cd data
bash download_dataset.sh
```

The challenge datasets are included in this code under `./data`

Then use the following command (in the root directory) to generate the few-shot data we need:

```bash
python tools/generate_k_shot_data.py
```

See `tools/generate_k_shot_data.py` for more options. For results in the paper, we use the default options: we take `K=16` and take 5 different seeds of 13, 21, 42, 87, 100. The few-shot data will be generated to `data/k-shot`. In the directory of each dataset, there will be folders named as `$K-$SEED` indicating different dataset samples.

**NOTE**: During training, the model will generate/load cache files in the data folder. If your data have changed, make sure to clean all the cache files (starting with "cache").

## Run

### Quick start
The code is built on [transformers](https://github.com/huggingface/transformers). We did all our experiments with
 version `3.4.0`.
 
Run the following bash file to reproduce the experiments:

```bash
bash run_experiment.sh
```

The script contain all commands for setting the experimentation, e.g., model (roberta-base/large), dataset, learning
 rate, num of epoch, etc.