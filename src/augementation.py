import numpy as np
from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
def augmentation(train_dataset : Dataset, ratio=0.8, maximum_length=768):
    def preprocess_left(example):
        # this preprocess will truncate the context from the left side

        answer_idx = example['answers']['answer_start'][0]

        example['context'] = example['context'][answer_idx:]
        example['answers']['answer_start'] = [0]

        return example

    def preprocess_right(example):
        # this preprocess will truncate the context from the right side

        answer_idx = example['answers']['answer_start'][0]

        example['context'] = example['context'][:answer_idx + len(example['answers']['text'][0])]
        return example

    def preprocess_middle(example):
        # this preprocess will truncate the context from the middle

        answer_idx = example['answers']['answer_start'][0]

        half_length = maximum_length // 2
        start = max(0, answer_idx - half_length)
        end = min(len(example['context']), answer_idx + half_length)

        example['context'] = example['context'][start:end]
        example['answers']['answer_start'] = [answer_idx - start]

        return example

    left_dataset_choice = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio))
    right_dataset_choice = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio))
    middle_dataset_choice = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio))

    dataset_l = train_dataset.select(left_dataset_choice).map(preprocess_left, num_proc=4)
    dataset_r = train_dataset.select(right_dataset_choice).map(preprocess_right, num_proc=4)
    dataset_m = train_dataset.select(middle_dataset_choice).map(preprocess_middle, num_proc=4)

    return concatenate_datasets([dataset_l, dataset_r, dataset_m])

def validate(train_dataset : Dataset):
    def preprocess(example):
        # this function will validate each train dataset example
        # if the answer is not in the context, we will remove the example
        answer = example['answers']['text'][0]
        answer_idx = example['answers']['answer_start'][0]

        if answer not in example['context']:
            example['id'] = "-1"
            return example

        if answer != example['context'][answer_idx:answer_idx + len(answer)]:
            example['id'] = "-1"
            return example

        return example

    train_dataset = train_dataset.map(preprocess, num_proc=4)
    train_dataset = train_dataset.filter(lambda example: example['id'] != "-1")

    return train_dataset

if __name__ == '__main__':
    import os
    parent_dir = os.path.dirname(os.getcwd())

    dataset = load_from_disk(parent_dir + '/data/train_dataset')
    augmented_dataset = augmentation(dataset['train'])
    print(augmented_dataset)

    augmented_dataset = validate(augmented_dataset)
    print(augmented_dataset)

    dataset['train'] = augmented_dataset
    dataset['train'] = dataset['train'].shuffle()
    dataset['validation'] = dataset['validation'].shuffle()

    # save the augmented dataset
    dataset.save_to_disk(parent_dir + '/data/AED_dataset')