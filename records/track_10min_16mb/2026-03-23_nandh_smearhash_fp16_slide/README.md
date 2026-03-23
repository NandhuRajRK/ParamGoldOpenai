This run adds a lightweight local-context path on top of the fp16-export + sliding-eval trainer:
- `SmearGate`: blends each token embedding with the previous token embedding
- `BigramHash`: hashed token-pair features projected into model space
- reduced `MLP_HIDDEN=960` to make room for the extra local features

Recommended command:

```bash
RUN_ID=nandh_exp2_smearhash_fp16_slide \
DATA_PATH=../../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=1024 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=256 \
MLP_HIDDEN=960 \
BIGRAM_HASH_BUCKETS=4096 \
BIGRAM_HASH_DIM=64 \
SMEARGATE=1 \
SMEARGATE_INIT=2.0 \
WARMDOWN_ITERS=3600 \
MATRIX_LR=0.055 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Why this run exists:
- The public SOTA moved toward explicit local token-pair features.
- This is the cheapest version of that idea that still fits the older int8 export path.
