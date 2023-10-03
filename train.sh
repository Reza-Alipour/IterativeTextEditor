#!/bin/bash

if [ "$1" == "RL" ]; then
    shift
    echo "Not Implemented" "$@"
elif [ "$1" == "PA" ]; then
    shift
    deepspeed FineTuneT5WithParallelData.py \
        --model_name google/t5-v1_1-large \
        --model_save_name t5-large-parallel-data-v0"$3" \
        --model_read_token "$HF_READ_TOKEN" \
        --model_write_token "$HF_WRITE_TOKEN" \
        --max_source_length 512 \
        --max_target_length 512 \
        --preprocessing_num_workers 4 \
        --dataset_read_token "$HF_READ_TOKEN" \
        --evaluation_strategy steps \
        --eval_steps 2000 \
        --save_strategy epoch \
        --do_train \
        --do_eval \
        --learning_rate 1e-4 \
        --gradient_accumulation_steps 1 \
        --overwrite_output_dir \
        --num_train_epochs 1 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --output_dir output/ \
        --deepspeed configs/ds_config.json


else
    echo "Invalid command. Usage: $0 {RL|PA} [args...]"
    exit 1
fi
