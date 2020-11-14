#!/bin/sh

python run_squad.py \
    --model_type bert \
    --model_name_or_path bert-pt/ \
    --output_dir squad-pt-small-saida/ \
    --data_dir squad-pt-small/ \
    --overwrite_output_dir \
    --do_train \
    --no_cuda \
    --train_file squad-train-v1.1.json \
    --do_lower_case \
    --do_eval \
    --predict_file squad-dev-v1.1.json \
    --per_gpu_train_batch_size 2 \
    --learning_rate 3e-5 \
    --num_train_epochs 2.0 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --threads 10 \
    --save_steps 5000