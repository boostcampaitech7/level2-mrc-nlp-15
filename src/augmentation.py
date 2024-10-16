import datasets
import random
import numpy as np
import copy
import time
from transformers import PreTrainedTokenizer
def augmentation(train_dataset : datasets.Dataset, tokenizer : dict):
    train_dataset_ = copy.deepcopy(train_dataset)
    #train_dataset_ = random_truncation(train_dataset_, )
    train_dataset_ = AEDA(train_dataset_, tokenizer, )

    # save augmented dataset
    timestamp = time.time()
    train_dataset.save_to_disk(f"../EDA/Train/train_augmented/{time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp))}.arrow")

    return train_dataset_

def random_truncation(train_dataset : datasets.Dataset, ratio=0.65, shred=0.44):
    if shred > 1.0:
        raise ValueError("shred must be less than 1.0")

    choice = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio) if ratio < 1 else ratio)
    train_dataset_ = train_dataset.select(choice)

    def preprocess_random_truncation(example, id_start):
        # ID 및 인덱스 설정
        example['id'] = f"mrc-id-0-{id_start}"
        example['__index_level_0__'] = id_start

        # 원래 context 길이
        original_len = len(example['context'])
        sample_start = example['answers']['answer_start'][0]

        # context의 일정 비율을 자르기 위해 shred 비율 설정 (예: 0.2)
        shred = 0.2
        shred_len = int(original_len * shred)

        # 왼쪽 및 오른쪽에서 잘라낼 수 있는 최대/최소 인덱스 계산
        minimum_left_truncated_index = min(sample_start, shred_len)
        maximum_right_truncated_index = min(
            original_len - len(example['answers']['text'][0]), shred_len
        )

        # 왼쪽 및 오른쪽에서 자를 인덱스 랜덤 선택
        truncated_left = random.randint(0, minimum_left_truncated_index)
        truncated_right = random.randint(0, maximum_right_truncated_index)

        # context를 잘라내고 answer_start 위치 조정
        example['context'] = example['context'][truncated_left:original_len - truncated_right]
        example['answers']['answer_start'][0] -= truncated_left

        return example

    train_dataset_ = train_dataset_.map(
        preprocess_random_truncation,
        with_indices=True,
    )

    return datasets.concatenate_datasets([train_dataset, train_dataset_])

def AEDA(train_dataset : datasets.Dataset, tokenizer : dict, ratio=0.33, min_puncation=3, max_puncation=6):
    random_idx = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio) if ratio < 1 else ratio)
    train_dataset_ = train_dataset.select(random_idx)

    # print(train_dataset_[:1]['context'])

    def preprocess_random_truncation(example, id_start):
        punctuation_list = ['.', ',', '!', '?', ':', ';']
        example = train_dataset_[0]
        id_start = 1000000

        # ID 및 인덱스 설정
        example['id'] = f"mrc-id-0-{id_start}"
        example['__index_level_0__'] = id_start

        # 원래 context 길이
        original_len = len(example['context'])
        sample_start = example['answers']['answer_start'][0]
        sample_end = sample_start + len(example['answers']['text'][0])

        num_of_punctuation = random.randint(min_puncation, max_puncation)

        for i in range(num_of_punctuation):
            punctuation = random.choice(punctuation_list)

            # select random index of context to insert punctuation
            insert_idx = random.randint(0, original_len)
            example['context'] = example['context'][:insert_idx] + punctuation + example['context'][insert_idx:]

            # we need +1 start index if punctuation is inserted before the answer_start
            if insert_idx <= sample_start:
                example['answers']['answer_start'][0] += 1

        return example

    train_dataset_ = train_dataset_.map(
        preprocess_random_truncation,
        with_indices=True,
    )

    # print()
    # print(train_dataset_[:1]['context'])

    return datasets.concatenate_datasets([train_dataset, train_dataset_])

def analysis():
    validation_dataset_dir = "../data/train_dataset/validation/dataset.arrow"
    validation_dataset_output_dir = "../EDA/Validation/uomnf97-klue-roberta-finetuned-korquad-v2_20241014_181927.json"
    n_best_dir = "../EDA/Validation/CurtisJeon-klue-roberta-large-korquad_v1_qa_20241015_101011_nbest_predictions.json"

    # print validation_dataset_output info
    validation_dataset_ = datasets.load_dataset('json', data_files=validation_dataset_output_dir)['train'][0]

    output_json = []
    for k, v in validation_dataset_.items():
        output_json.append({
            k: v
        })

    # print arrow file as json
    validation_dataset = datasets.Dataset.from_file(validation_dataset_dir)

    # make new json only with 'id', and 'answers' : 'text'
    answer_json = []
    for sample in validation_dataset:
        answer_json.append({
            sample['id']: sample['answers']['text'][0]
        })

    # fine diff
    for output, answer in zip(output_json, answer_json):
        if output != answer:
            print(f"id : {str(list(output.keys())[0])}, output : {str(list(output.values())[0])} != answer : {str(list(answer.values())[0])}")
            print()

if __name__ == "__main__":
    train_dataset_dir = "../data/train_dataset/train/dataset.arrow"
    train_dataset = datasets.Dataset.from_file(train_dataset_dir)

    train_dataset = augmentation(train_dataset, tokenizer={})
    #analysis()