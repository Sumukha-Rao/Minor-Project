---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: label
    dtype: int64
  splits:
  - name: train
    num_bytes: 17982666
    num_examples: 50000
  - name: test
    num_bytes: 3551249
    num_examples: 10000
  download_size: 13293148
  dataset_size: 21533915
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
