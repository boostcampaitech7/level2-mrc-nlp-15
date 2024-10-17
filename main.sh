# #!/bin/bash

current_time=$(date -u -d "+9 hours" "+%Y%m%d_%H%M%S")

# Set up directories
train_dir="/data/ephemeral/home/level2-mrc-nlp-15/models/train_${current_time}"
predict_dir="/data/ephemeral/home/level2-mrc-nlp-15/output/test_${current_time}"
predict_dataset_name="/data/ephemeral/home/level2-mrc-nlp-15/data/test_dataset"
# train_dataset_name="/data/ephemeral/home/level2-mrc-nlp-15/data/train_dataset"

cd /data/ephemeral/home/level2-mrc-nlp-15/src

# Perform training
# python main.py --output_dir $train_dir --do_train
# python main.py --output_dir ./models/klue-roberta-korquad --do_train --model_name_or_path uomnf97/klue-roberta-finetuned-korquad-v2 --dataset_name ./data/train_dataset --overwrite_output_dir --max_seq_length 512 --per_device_train_batch_size 16 --learning_rate 1e-5 --num_train_epochs 3
# python main.py --output_dir $train_dir --do_train
python main.py --output_dir $train_dir --do_train --overwrite_output_dir --per_device_train_batch_size 16 --learning_rate 1e-5 --num_train_epochs 3

# # Perform evaluation (optional)
eval_dir="/data/ephemeral/home/level2-mrc-nlp-15/output/train_dataset_${current_time}"
#python main.py --output_dir $eval_dir --do_eval
# python main.py --output_dir ./outputs/train_dataset --model_name_or_path ./models/train_dataset/ --do_eval 
# python main.py --output_dir $eval_dir --do_eval
python main.py --output_dir $eval_dir --model_name_or_path $train_dir --do_eval

# Perform prediction (inference)
# python inference_.py --output_dir $predict_dir --dataset_name $predict_dataset_name --model_name_or_path $train_dir --do_predict
python main.py --output_dir $predict_dir --dataset_name $predict_dataset_name --model_name_or_path $train_dir --do_predict
# python src/main.py --output_dir /data/ephemeral/home/level2-mrc-nlp-15/output --dataset_name /data/ephemeral/home/level2-mrc-nlp-15/data/test_dataset --model_name_or_path /data/ephemeral/home/level2-mrc-nlp-15/models/train_20241017_101325 --do_predict

# Print Done
echo "All Done. Check the output in ${predict_dir}"
