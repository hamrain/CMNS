#!/bin/bash

NUM_GPU=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1

# template for bert, originate from PromptBERT
TEMPLATE="*cls*_This_sentence_of_\"*sent_0*\"_means*mask*.*sep+*"
TEMPLATE2="*cls*_This_sentence_:_\"*sent_0*\"_means*mask*.*sep+*"


python -m torch.distributed.launch --nproc_per_node $NUM_GPU --master_port 14481  train_prompt.py \
    --model_name_or_path ./bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/CMNS-pro-bert \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --save_steps 125 \
    --save_total_limit 1 \
    --learning_rate 5e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --mlp_only_train \
    --overwrite_output_dir \
    --do_train \
    --kmeans 96 \
    --kmean_cosine 0.4 \
    --mask_embedding_sentence \
    --mask_embedding_sentence_delta \
    --mask_embedding_sentence_template $TEMPLATE\
    --mask_embedding_sentence_different_template $TEMPLATE2\
    --fp16 \
    --bml_weight 1e-5 \
    --bml_alpha 0.26 \
    --bml_beta  0.10 \
    --guss_weight 0.27 \
    --early_stop 5 \
    "$@"

python evaluation_prompt.py \
--model_name_or_path result/CMNS-pro-bert \
--pooler avg \
--task_set sts \
--mask_embedding_sentence \
--mask_embedding_sentence_template $TEMPLATE \
--mode test
