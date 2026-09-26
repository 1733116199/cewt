#!/bin/bash

export W_QUANT_KWARGS="{\"bits\": ${1}}"
export A_QUANT_KWARGS="{\"bits\": ${1}}"

NAME="v51-imnet64${4}-A${3}-WQuESTTGCS-${1}-constraint_${2}"

torchrun --nproc_per_node=1 main.py \
    --model deit_base_patch16_224 \
    --output_dir checkpoints/$NAME \
    --wandb_name $NAME \
    --data-set IMNET64 \
    --epochs ${4} \
    --warmup-epochs 5 \
    --hyperball_optimization 1 \
    --hyperball_constraint ${2} \
    --matrix_lr 0.004 \
    --batch-size 1024 \
    --input-size 64 \
    --lr 2.5e-4 \
    --num_workers 4 \
    --sched linear \
    --print-freq 100 \
    --w-quant QuESTTGCS \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${3} \
    --a-quant-kwargs "${A_QUANT_KWARGS}"
    