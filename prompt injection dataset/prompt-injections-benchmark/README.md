---
license: cc-by-nc-4.0
tags:
- prompt
- injection
- jailbreak
- benign
dataset_info:
  features:
  - name: text
    dtype: string
  - name: label
    dtype: string
  splits:
  - name: test
    num_bytes: 3637463
    num_examples: 5000
  download_size: 2109708
  dataset_size: 3637463
configs:
- config_name: default
  data_files:
  - split: test
    path: data/test-*
---

# Dataset: Qualifire Benchmark Prompt Injection(Jailbreak vs. Benign) Datasets

## Overview
This dataset contains **5,000 prompts**, each labeled as either `jailbreak` or `benign`. The dataset is designed for evaluating AI models' robustness against adversarial prompts and their ability to distinguish between safe and unsafe inputs.

## Dataset Structure
- **Total Samples:** 5,000
- **Labels:** `jailbreak`, `benign`
- **Columns:** 
  - `text`: The input text
  - `label`: The classification (`jailbreak` or `benign`)


## License
cc-by-nc-4.0
