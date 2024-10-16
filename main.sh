#!/bin/bash
current_time=$(date -u -d "+9 hours" "+%Y%m%d_%H%M%S")
# Set up directories
model_name_or_path="CurtisJeon/klue-roberta-large-korquad_v1_qa"
#model_name_or_path="uomnf97/klue-roberta-finetuned-korquad-v2"
train_dir="models/train_${current_time}"
#train_dir="models/train_20241015_173117"
eval_dir="output/eval_${current_time}"
predict_dir="output/test_${current_time}"
predict_dataset_name="data/test_dataset"
augmentation="True"

#cd /data/ephemeral/home/level2-mrc-nlp-15/src
# Perform training
python src/main.py --output_dir $train_dir --model_name_or_path $model_name_or_path --do_train --max_seq_length 384 --per_device_train_batch_size 16 --num_train_epochs 3 --learning_rate "1e-5" --augmentation $augmentation

## # Perform evaluation (optional)
#python src/main.py --output_dir $train_dir --do_eval --model_name_or_path $train_dir
#
## Perform prediction (inference)
#python src/main.py --output_dir $predict_dir --dataset_name $predict_dataset_name --model_name_or_path $train_dir --do_predict
#
## Print Done
#echo "All Done. Check the output in ${predict_dir}"