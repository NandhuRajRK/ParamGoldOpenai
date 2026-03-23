This run combines three early public wins in the repo into one stronger proxy:
- fp16 tied embedding passthrough during quantized export
- `TRAIN_SEQ_LEN=2048`
- sliding-window final evaluation

Recommended command:

```bash
RUN_ID=nandh_exp1_fp16_slide_seq2048 \
DATA_PATH=../../../../data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=../../../../data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TRAIN_SEQ_LEN=2048 \
EVAL_STRIDE=64 \
EVAL_BATCH_SEQS=128 \
MLP_HIDDEN=992 \
WARMDOWN_ITERS=3600 \
MATRIX_LR=0.06 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Why this run exists:
- It is the lowest-risk combination based on already-public ideas in the repo.
- It keeps the model nearly baseline-sized while improving both training context and final evaluation context.
