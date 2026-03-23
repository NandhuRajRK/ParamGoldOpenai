This is the most aggressive of the three proxy runs:
- local-context path via `SmearGate + BigramHash`
- `TRAIN_SEQ_LEN=2048`
- sliding-window final evaluation
- slightly smaller `MLP_HIDDEN=928` to offset the larger local-feature projection

Recommended command:

```bash
RUN_ID=nandh_exp3_smearhash_fp16_slide_seq2048 \
DATA_PATH=../../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=128 \
MLP_HIDDEN=928 \
BIGRAM_HASH_BUCKETS=4096 \
BIGRAM_HASH_DIM=96 \
SMEARGATE=1 \
SMEARGATE_INIT=2.0 \
TIED_EMBED_LR=0.04 \
WARMDOWN_ITERS=3600 \
MATRIX_LR=0.05 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Why this run exists:
- It tests whether the public long-context gain and the local-feature gain stack cleanly.
- If this variant is too large, reduce `MLP_HIDDEN` before changing anything else.
