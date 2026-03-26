# RunPod Commands

## 1. Baseline

Run this once on the same hardware you will use for the SOTA stack.

```bash
cd /workspace/parameter-golf
RUN_ID=baseline_sp1024 \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

## 2. SOTA-Chasing Stack

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-26_nandh_sota_stack
RUN_ID=nandh_sota_stack \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

## 3. Safer Fallback If Compile Misbehaves

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-26_nandh_sota_stack
RUN_ID=nandh_sota_stack_nocompile \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
USE_TORCH_COMPILE=0 \
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

## 4. If `zstandard` Is Missing

```bash
pip install zstandard
```

Or run with:

```bash
USE_ZSTD=0 python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```
