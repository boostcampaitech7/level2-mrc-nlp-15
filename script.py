import subprocess
import os
import re
from datetime import datetime

def run_command(command):
    try:
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error in executing the command:", e)

def extract_timestamp(dir_name):
    match = re.search(r'_(\d{8}_\d{4})$', dir_name)
    if match:
        timestamp_str = match.group(1)
        try:
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
            return timestamp
        except ValueError:
            return None
    return None

def find_model_dir(timestamp=None):
    models_root = os.path.join(os.sep, 'data', 'ephemeral', 'home', 'level2-mrc-nlp-15', 'models')
    model_dir = None

    if timestamp:
        model_dirs = [os.path.join(models_root, d) for d in os.listdir(models_root)
                      if os.path.isdir(os.path.join(models_root, d))]
        print("model_dirs:", model_dirs)
        for dir_path in model_dirs:
            dir_name = os.path.basename(dir_path)
            dir_timestamp = extract_timestamp(dir_name)
            if dir_timestamp and dir_timestamp.strftime("%Y%m%d_%H%M") == timestamp:
                model_dir = dir_path
                break
        if model_dir and os.path.isdir(model_dir):
            print(f"Using model directory: {model_dir}")
            return model_dir
        else:
            print(f"No model directory found with timestamp {timestamp}.")
            return None
    else:
        # Find the latest model directory based on timestamps in directory names
        model_dirs = [os.path.join(models_root, d) for d in os.listdir(models_root)
                      if os.path.isdir(os.path.join(models_root, d))]
        if not model_dirs:
            print("No model directories found.")
            return None

        # Extract timestamps and filter valid directories
        model_dirs_with_timestamps = []
        for dir_path in model_dirs:
            dir_name = os.path.basename(dir_path)
            timestamp = extract_timestamp(dir_name)
            if timestamp:
                model_dirs_with_timestamps.append((dir_path, timestamp))
            else:
                print(f"Skipping directory {dir_name}, invalid timestamp format.")

        if not model_dirs_with_timestamps:
            print("No valid model directories found.")
            return None

        # Find the directory with the latest timestamp
        latest_model_dir = max(model_dirs_with_timestamps, key=lambda x: x[1])[0]

        if os.path.isdir(latest_model_dir):
            print(f"Using latest model directory: {latest_model_dir}")
            return latest_model_dir
        else:
            print(f"Could not find a suitable model directory.")
            return None

def main(timestamp=None):
    base_directory = os.path.join(os.sep, 'data', 'ephemeral', 'home', 'level2-mrc-nlp-15')
    os.chdir(os.path.join(base_directory, 'src'))

    model_dir = None  # model_dir 변수를 초기화합니다.

    if not timestamp:
            print("Training a new model.")

            # Training and Evaluation
            run_command([
                'python', 'main.py',
                '--output_dir', os.path.join(base_directory, 'models'),
                '--do_train'
            ])
            model_dir = find_model_dir()  # Find the latest model directory
            if not model_dir:
                # Could not find model directory with the provided timestamp
                print("Error: Could not find a suitable model directory.")
                return
    else:
        model_dir = find_model_dir(timestamp)

    # Evaluation
    run_command([
        'python', 'main.py',
        '--output_dir', os.path.join(base_directory, 'output'),
        '--model_name_or_path', model_dir,
        '--do_eval'
    ])

    # Prediction
    run_command([
        'python', 'main.py',
        '--output_dir', os.path.join(base_directory, 'output'),
        '--dataset_name', os.path.join(base_directory, 'data', 'test_dataset'),
        '--model_name_or_path', model_dir,
        '--do_predict'
    ])

    print(f"All Done. Check the output in {os.path.join(base_directory, 'output')} directory.")

if __name__ == "__main__":
    timestamp_input = input("Enter timestamp if you want to use a specific model directory or press enter to train a new model >> ")
    main(timestamp_input.strip() if timestamp_input else None)