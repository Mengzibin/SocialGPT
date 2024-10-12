#!/bin/bash

export WANDB_MODE=disabled

export model=$1 # llama2 or vicuna

# Create results folder if it doesn't exist
if [ ! -d "../results" ]; then
    mkdir "../results"
    echo "Folder '../results' created."
else
    echo "Folder '../results' already exists."
fi

cd ../

CUDA_VISIBLE_DEVICES=0 python -u main.py \
    --config="configs/transfer_${model}.py" \
    --config.attack=gcg \
    --config.result_prefix="results/transfer_${model}_gcg_progressive" \
    --config.progressive_goals=False \
    --config.stop_on_success=True \
    --config.num_train_models=1 \
    --config.allow_non_ascii=True \
    --config.n_steps=10 \
    --config.test_steps=1 \
    --config.batch_size=12 \
    --config.image_txt="PISC/relation_split/relation_trainidx.txt" \
    --config.image_txt_test="PISC/relation_split/relation_testidx.txt" \
    --config.sam4story="dataset-pisc" \
    --config.example_dir="definition_of_example" \
    --config.relationship_dir="definition_of_relationship" \
    --config.prelude_dir="prelude" \
    --config.story_dir="story_train" \
    --config.system_prompt_dir="system_prompt" \
    --config.story_dir_test="story_test" \
    --config.filter_cand=False

