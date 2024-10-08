import train
import inference
import yaml
import time

if __name__ == '__main__':
    # read config yaml and get output dir
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # set output_dir as --output_dir in parser
    model_output_dir = config['path']['model_path']
    model_output_dir += time.strftime("%Y%m%d_%H%M%S", time.localtime())

    test_output_dir = config['path']['output_path']
    test_dataset = config['path']['test_path']

    args = {
        'model_path': model_output_dir,
        'output_path': test_output_dir,
        'test_path': test_dataset
    }

    train.main(args)
    inference.main(args)