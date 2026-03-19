This folder is a personal starter workspace for a Parameter Golf submission.

Starting point:
- Based on `2026-03-18_FP16Embed_WD3600`
- Reason: it is materially better than the naive baseline, but much simpler to reason about than the sliding-window eval submission

Current recipe:
- Keep `tok_emb.weight` in `fp16` during post-training serialization
- Use `MLP_HIDDEN=992` to stay under the `16,000,000` byte cap
- Use `WARMDOWN_ITERS=3600`
- Use `MATRIX_LR=0.06`

Suggested first run on `8xH100`:

```bash
RUN_ID=nandh_fp16embed_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MLP_HIDDEN=992 \
WARMDOWN_ITERS=3600 \
MATRIX_LR=0.06 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

What to change first:
- `TRAIN_SEQ_LEN=2048` if the box still stays within the time budget
- `TIED_EMBED_LR` in the `0.04` to `0.06` range
- `MATRIX_LR` in the `0.05` to `0.07` range
- `WARMDOWN_ITERS` in the `2400` to `4200` range
- `SEED` across at least 3 runs before claiming a record

Submission checklist:
- Keep exact train logs for every reported run
- Fill in `submission.json` with your metadata and final metrics
- Make sure this folder is standalone and runnable from inside the folder
- Confirm printed `Total submission size int8+zlib` is below `16000000`
- If you change tokenizer or eval, justify the `val_bpb` calculation carefully

Practical note:
- This local Windows workspace currently does not have the dataset or Python packages installed.
- For real runs, use a remote CUDA machine such as RunPod and copy this folder there.
