#!/bin/bash

current_time=$(date -u -d "+9 hours" "+%Y%m%d_%H%M%S")

# Set up directories
train_dir="/data/ephemeral/home/level2-mrc-nlp-15/models/train_${current_time}"
predict_dir="/data/ephemeral/home/level2-mrc-nlp-15/output/test_${current_time}"
predict_dataset_name="/data/ephemeral/home/level2-mrc-nlp-15/data/test_dataset"

cd /data/ephemeral/home/level2-mrc-nlp-15/src

# Perform training
python main.py --output_dir $train_dir --do_train

# # Perform evaluation (optional)
# eval_dir="/data/ephemeral/home/level2-mrc-nlp-15/output/train_dataset_${current_time}"
# python main.py --output_dir $eval_dir --do_eval

# Perform prediction (inference)
python main.py --output_dir $predict_dir --dataset_name $predict_dataset_name --model_name_or_path $train_dir --do_predict

# Print Done
echo "All Done. Check the output in ${predict_dir}"