# SOTA-Chasing Mixed Int5/Int6 + BigramHash + SmearGate + SWA

This record folder is a runnable scaffold for a current public-leaderboard style Parameter Golf attack. It is not a finished record submission yet; the metrics in `submission.json` must be filled after real runs.

## What Is Implemented

1. `10` layers by default, `512` model dim, `8` heads, `4` KV heads
2. `3x` MLP by default (`MLP_MULT=3`)
3. `BigramHash` enabled by default with `10240` buckets and `128`-dim projection
4. `SmearGate` enabled by default
5. Orthogonal init for large matrices, with smaller gain on residual projections
6. Decoupled weight decay:
   `MUON_WEIGHT_DECAY=0.04`, `ADAM_WEIGHT_DECAY=0.01`
7. SWA snapshots over the late portion of training
8. Mixed export:
   MLP matrices default to `int5`, attention/other large matrices default to `int6`
9. `tok_emb.weight` kept in fp16 by default
10. Final layer key projection kept in fp16 by default
11. `zstd` compression by default, with `zlib` fallback
12. Sliding-window final evaluation enabled by default (`EVAL_STRIDE=64`)

## Default Run Intent

The defaults aim at challenge-style hardware:

- `TRAIN_SEQ_LEN=2048`
- `TRAIN_BATCH_TOKENS=786432`
- `NUM_LAYERS=10`
- `MLP_MULT=3`
- `MATRIX_LR=0.02`
- `SCALAR_LR=0.02`
- `TIED_EMBED_LR=0.03`
- `MUON_MOMENTUM=0.99`
- `MUON_MOMENTUM_WARMUP_START=0.92`
- `MUON_MOMENTUM_WARMUP_STEPS=1500`
- `WARMDOWN_ITERS=3000`
- `SWA_START_FRAC=0.5`
- `SWA_EVERY=50`

## First Run

On RunPod / challenge-style CUDA machines:

```bash
cd /workspace/parameter-golf/records/track_10min_16mb/2026-03-26_nandh_sota_stack
RUN_ID=nandh_sota_stack \
DATA_PATH=/workspace/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

If `torch.compile` is unstable on the first paid run:

```bash
USE_TORCH_COMPILE=0 python -m torch.distributed.run --standalone --nproc_per_node=8 train_gpt.py
```

## Dependencies

`zstandard` is optional but recommended. If absent, set `USE_ZSTD=0`.

