#!/bin/bash
CONFIG_FILE="configs/deim_dfine/object365/dfine_hgnetv2_l_custom.yml"
OUTPUT_DIR="output/l_custom"
OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=4 CUDA_VISIBLE_DEVICES=0,1,2 torchrun --master_port=7777 --nproc_per_node=3 train.py -c $CONFIG_FILE --use-amp --seed=0 --output-dir $OUTPUT_DIR -r ${OUTPUT_DIR}/last.pth