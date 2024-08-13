##!/bin/bash
#
NUM_GPU=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0

python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port 13212 train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/CMNS-nonpro-bert \
    --num_train_epochs 1\
    --per_device_train_batch_size 256 \
    --save_steps 125 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --do_train \
    --kmeans 128\
    --kmean_cosine 0.4 \
    --bml_weight 1e-4 \
    --bml_alpha 0.20 \
    --bml_beta 0.00 \
    --guss_weight 0.125\
    --early_stop 6 \
    --fp16 \
    "$@"


#python evaluation.py \
#--model_name_or_path result/CMNS-nonpro-bert \
#--pooler cls_before_pooler \
#--task_set sts \
#--mode test


