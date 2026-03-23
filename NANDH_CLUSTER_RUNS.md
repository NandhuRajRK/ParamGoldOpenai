# Parameter Golf Experiment Runs

These runs are intended as fast proxy experiments on the cluster.

Important:
- Official leaderboard target is still `8xH100` with `10` minutes of training and under `10` minutes of evaluation.
- Runs on `1xA40` or `1xA100` are useful for ranking ideas, not for claiming a record.

## One-Time Cluster Setup

```bash
cd /vol/miltank/users/rajn/repos/ParamGoldOpenai
git pull

tmux new -s paramgolf
salloc --partition=universe --time=0-02:00:00 --gres=gpu:1,gpumem:40G --cpus-per-task=8 --mem=32G

module load python/anaconda3
conda activate paramgolf || conda create -n paramgolf python=3.12 -y && conda activate paramgolf
pip install -r requirements.txt

# Full challenge cache once:
python data/cached_challenge_fineweb.py --variant sp1024
```

## Experiment 1

Folder:

```bash
cd /vol/miltank/users/rajn/repos/ParamGoldOpenai/records/track_10min_16mb/2026-03-23_nandh_fp16_slide_seq2048
```

Command:

```bash
export RUN_ID=nandh_exp1_fp16_slide_seq2048
export DATA_PATH=../../../../data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=../../../../data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export TRAIN_SEQ_LEN=2048
export EVAL_STRIDE=64
export EVAL_BATCH_SEQS=128
export MLP_HIDDEN=992
export WARMDOWN_ITERS=3600
export MATRIX_LR=0.06
export MAX_WALLCLOCK_SECONDS=600
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Experiment 2

Folder:

```bash
cd /vol/miltank/users/rajn/repos/ParamGoldOpenai/records/track_10min_16mb/2026-03-23_nandh_smearhash_fp16_slide
```

Command:

```bash
export RUN_ID=nandh_exp2_smearhash_fp16_slide
export DATA_PATH=../../../../data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=../../../../data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export TRAIN_SEQ_LEN=1024
export EVAL_STRIDE=64
export EVAL_BATCH_SEQS=256
export MLP_HIDDEN=960
export BIGRAM_HASH_BUCKETS=4096
export BIGRAM_HASH_DIM=64
export SMEARGATE=1
export SMEARGATE_INIT=2.0
export WARMDOWN_ITERS=3600
export MATRIX_LR=0.055
export MAX_WALLCLOCK_SECONDS=600
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Experiment 3

Folder:

```bash
cd /vol/miltank/users/rajn/repos/ParamGoldOpenai/records/track_10min_16mb/2026-03-23_nandh_smearhash_fp16_slide_seq2048
```

Command:

```bash
export RUN_ID=nandh_exp3_smearhash_fp16_slide_seq2048
export DATA_PATH=../../../../data/datasets/fineweb10B_sp1024
export TOKENIZER_PATH=../../../../data/tokenizers/fineweb_1024_bpe.model
export VOCAB_SIZE=1024
export TRAIN_SEQ_LEN=2048
export EVAL_STRIDE=64
export EVAL_BATCH_SEQS=128
export MLP_HIDDEN=928
export BIGRAM_HASH_BUCKETS=4096
export BIGRAM_HASH_DIM=96
export SMEARGATE=1
export SMEARGATE_INIT=2.0
export TIED_EMBED_LR=0.04
export WARMDOWN_ITERS=3600
export MATRIX_LR=0.05
export MAX_WALLCLOCK_SECONDS=600
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Quick Triage

After each run, check:
- `final_int8_zlib_roundtrip_exact`
- `Total submission size int8+zlib`
- `peak memory allocated`

If any run exceeds `16000000` bytes, cut `MLP_HIDDEN` first.
