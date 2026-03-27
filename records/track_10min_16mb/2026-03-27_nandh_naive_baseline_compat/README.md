## Naive Baseline Compat Control

This folder is a runtime-compatibility control derived from the public `2026-03-17_NaiveBaseline` record.

It is intended only to check whether the current pod environment produces numbers in the same ballpark as the published baseline.

## Compatibility-Only Changes

- replace `torch.inference_mode()` with `torch.no_grad()` in validation
- replace `enable_gqa=...` with explicit KV expansion before SDPA
- add `USE_TORCH_COMPILE=0/1` toggle

These changes are meant to preserve the same ML behavior while allowing the baseline to run on pod images with:

- older PyTorch SDPA signatures
- runtime issues around inference-mode tensors
- unstable `torch.compile`

## Purpose

This is **not** a new method. It is only a control run to sanity-check the H100 environment against the published baseline family.

## Run

```bash
cd /workspace/ParamGoldOpenAI/records/track_10min_16mb/2026-03-27_nandh_naive_baseline_compat
RUN_ID=naive_baseline_compat_h100x8 \
DATA_PATH=/workspace/ParamGoldOpenAI/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/ParamGoldOpenAI/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
USE_TORCH_COMPILE=0 \
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```
