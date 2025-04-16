#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port=7777 --nproc_per_node=3 train.py -c configs/deim_dfine/deim_hgnetv2_l_custom.yml --use-amp --seed=0
