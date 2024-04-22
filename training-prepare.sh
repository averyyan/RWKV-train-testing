#!/bin/bash
MODEL_TYPE="x060"
N_LAYER="12"
N_EMBD="768"
CTX_LEN="512"
PROJ_DIR="out/L"$N_LAYER"-D"$N_EMBD"-"$MODEL_TYPE"-"$CTX_LEN
python3 train.py --wandb "" --proj_dir $PROJ_DIR \
 --data_file "data/minipile" --data_type "binidx" --vocab_size 65536 --my_testing $MODEL_TYPE \
 --ctx_len $CTX_LEN --my_pile_stage 1 --epoch_count 1 --epoch_begin 0 \
 --epoch_save 1 --weight_decay 0 --head_size_a 64 \
 --num_nodes 1 --micro_bsz 1 --n_layer $N_LAYER --n_embd $N_EMBD --pre_ffn 0 --head_qk 0 --my_exit_tokens 1498226207 --magic_prime 2926181 \
 --lr_init 1e-5 --lr_final 1e-5 --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 --my_pile_edecay 0 \
 --accelerator cpu --devices 1 --precision bf16 --strategy deepspeed_stage_2 --grad_cp 1
