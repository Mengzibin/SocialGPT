#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py \
--image_folder='PISC/image' \
--image_txt='relation_trainidx.txt' \
--dir_result='result' \
--api_key='openai-key' \
--image_caption_device='cuda' \
--semantic_segment_device='cuda' \
--contolnet_device='cuda' \
--gpt_version='gpt-4' \
--sam_arch='vit_h' \
--captioner_base_model='blip2' \
--region_classify_model='edit_anything' \
--gpt_version_relation='gpt-4' \
--index=${1}
