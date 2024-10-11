import os
import yaml
import time

import main

os.path.join(os.path.dirname(__file__), 'config.yaml')

if __name__ == '__main__':
    # read config yaml and get output dir
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # set output_dir as --output_dir in parser
    model_name = config['model']['model_name'].replace('/', '-')

    model_dir = os.path.abspath(config['path']['model_path'])
    output_dir = os.path.abspath(config['path']['output_path'])
    test_dataset = os.path.abspath(config['path']['test_path'])

    model_output_dir = os.path.join(model_dir, model_name + time.strftime("_%Y%m%d_%H%M%S", time.localtime()))
    test_output_dir = os.path.join(output_dir, model_name + time.strftime("_%Y%m%d_%H%M%S", time.localtime()))

    args = {
        'model_path': model_output_dir,
        'output_path': test_output_dir,
        'test_path': test_dataset
    }

    # check model already trained in model path
    model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and model_name in d]

    if not model_dirs:
        print('No model found in model path. Training new model name : ', model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        main.main(args, do_train=True)

    else:
        # Ask user to train still
        print(f'Model {model_name} already trained. Do you want to train again? (y/n)')
        answer = input()
        while answer not in ['y', 'n']:
            print('Invalid answer. Please answer y or n.')
            answer = input()

        if answer == 'y':
            os.makedirs(model_output_dir, exist_ok=True)
            main.main(args, do_train=True)

        else:
            args['model_path'] = os.path.join(model_dir, model_dirs[0])
            args['output_path'] = os.path.join(output_dir, model_dirs[0])

    main.main(args, do_eval=True)
    main.main(args, do_predict=True)