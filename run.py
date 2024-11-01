#!/usr/bin/env python3

import os
import subprocess
from datetime import datetime, timedelta

# Get current time (UTC + 9 hours)
current_time = datetime.utcnow() + timedelta(hours=9)
current_time_str = current_time.strftime('%Y%m%d_%H%M%S')

# Root directory (adjust this if necessary)
root_dir = os.getcwd()
#root_dir = os.path.join(os.sep, 'data', 'ephemeral', 'home', 'level2-mrc-nlp-15')

# Ensure root directory exists
if not os.path.exists(root_dir):
    raise FileNotFoundError(f"The root directory {root_dir} does not exist. Please adjust the path accordingly.")

# Set up directories
train_dir = os.path.join(root_dir, 'models', f'train_{current_time_str}')
predict_dir = os.path.join(root_dir, 'output', f'test_{current_time_str}')
predict_dataset_name = os.path.join(root_dir, 'data', 'test_dataset')

# Change to src directory
src_dir = os.path.join(root_dir, 'src')
if not os.path.exists(src_dir):
    raise FileNotFoundError(f"The source directory {src_dir} does not exist. Please adjust the path accordingly.")
os.chdir(src_dir)

# Perform training
subprocess.run([
    "python", "main.py",
    "--output_dir", train_dir,
    "--do_train",
    "--overwrite_output_dir",
    "--per_device_train_batch_size", "16",
    "--learning_rate", "1e-5",
    "--num_train_epochs", "3"
], check=True)

# Perform evaluation (optional)
eval_dir = os.path.join(root_dir, 'output', f'train_dataset_{current_time_str}')
subprocess.run([
    "python", "main.py",
    "--output_dir", eval_dir,
    "--do_eval"
], check=True)

# Perform prediction (inference)
subprocess.run([
    "python", "main.py",
    "--output_dir", predict_dir,
    "--dataset_name", predict_dataset_name,
    "--model_name_or_path", train_dir,
    "--do_predict"
], check=True)

# Print Done
print(f"All Done. Check the output in {predict_dir}")