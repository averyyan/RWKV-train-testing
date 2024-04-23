#!/bin/bash

MODEL_TYPE="x060"
N_LAYER="12"
N_EMBD="768"
CTX_LEN="512" 
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE"-"$CTX_LEN
M_BSZ="8"
LR_INIT="6e-4"
LR_FINAL="6e-5"
GRAD_CP=1
EPOCH_SAVE=10 
N_NODE=1 
GPU_PER_NODE=1 
DS_BUCKET_MB=2

# python3 train.py --proj_dir $PROJ_DIR --my_testing $MODEL_TYPE \
#  --ctx_len $CTX_LEN --my_pile_stage 3 --epoch_count 999999 --epoch_begin 0 \
#  --data_file "data/minipile" --my_exit_tokens 1498226207 --magic_prime 2926181 \
#  --num_nodes $N_NODE --micro_bsz $M_BSZ --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 \
#  --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 --data_type "binidx" --vocab_size 65536 \
#  --weight_decay 0.001 --epoch_save $EPOCH_SAVE --head_size_a 64 \
#  --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --grad_cp $GRAD_CP --enable_progress_bar True --ds_bucket_mb $DS_BUCKET_MB \
#  --lora_load rwkv-0 --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln \
#  --PISSA --svd_niter 4 \
#  --quant nf4

python3 train.py --proj_dir $PROJ_DIR --data_file "data/minipile" \
    --data_type binidx --vocab_size 65536 \
    --ctx_len $CTX_LEN --epoch_steps 1000 --epoch_count 20 --epoch_begin 0 --epoch_save $EPOCH_SAVE --micro_bsz $M_BSZ \
    --n_layer $N_LAYER --n_embd $N_EMBD \
    --pre_ffn 0 --head_qk 0 --lr_init $LR_INIT --lr_final $LR_FINAL --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1 \
    --my_testing "x060" \
    --lora_load rwkv-0 --lora --lora_r 64 --lora_alpha 128 --lora_dropout 0.01 --lora_parts=att,ffn,time,ln \
    --PISSA --svd_niter 4 \
    --quant nf4