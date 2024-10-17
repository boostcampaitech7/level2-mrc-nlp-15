import os

import datasets
import random
import numpy as np
import copy
import time
import pandas as pd
import re
import nltk
from transformers import PreTrainedTokenizer
from konlpy.tag import Okt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def print_sample(train_dataset_):
    for k, v in train_dataset_[:1].items():
        print(f"{k} : {v}")
    answer_start = train_dataset_[:1]['answers'][0]['answer_start'][0]
    answer_end = answer_start + len(train_dataset_[:1]['answers'][0]['text'][0])
    print(f"answer : {train_dataset_[:1]['context'][0][answer_start:answer_end]}")

def augmentation(train_dataset : datasets.Dataset, tokenizer : dict):
    train_dataset_ = copy.deepcopy(train_dataset)

    #train_dataset_ = stop_word(train_dataset_, tokenizer, )
    train_dataset_ = random_truncation_all(train_dataset_, )
    #train_dataset_ = swap_sentence(train_dataset_, tokenizer, ratio=0.33)
    #train_dataset_ = random_truncation(train_dataset_, )
    train_dataset_ = AEDA(train_dataset_, tokenizer, )

    # save augmented dataset
    timestamp = time.time()
    train_dataset.save_to_disk(f"../EDA/Train/train_augmented/{time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp))}.arrow")

    return train_dataset_

def random_truncation(train_dataset : datasets.Dataset, ratio=0.3, shred=0.44, concat=True):
    if shred > 1.0:
        raise ValueError("shred must be less than 1.0")

    choice = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio) if ratio < 1 else ratio, replace=False)
    train_dataset_ = train_dataset.select(choice)

    left_choice = list(set(range(len(train_dataset))) - set(choice))
    train_dataset_left = train_dataset.select(left_choice)

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

    return datasets.concatenate_datasets([train_dataset, train_dataset_]) if concat else datasets.concatenate_datasets([train_dataset_left, train_dataset_])

def random_truncation_all(train_dataset : datasets.Dataset, ratio=0.75, concat=True):
    choice = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio) if ratio <= 1 else ratio, replace=False)
    train_dataset_ = train_dataset.select(choice)

    # get left data that are not in choice
    left_choice = list(set(range(len(train_dataset))) - set(choice))
    train_dataset_left = train_dataset.select(left_choice)

    #print_sample(train_dataset_)

    def preprocess(example, id_start):
        id_start = 1000000
        original_context = example['context']
        # ID 및 인덱스 설정
        example['id'] = f"mrc-id-0-{id_start}"
        example['__index_level_0__'] = id_start

        answer_start = example['answers']['answer_start'][0]

        context = example['context'][0:answer_start] + "[HERE]" + example['context'][answer_start:]
        answer_start_ = answer_start + len("[HERE]")

        # trim context
        context = re.sub(r'\\n', '', context)

        context_split = context.split('다.')
        context_split.pop() if context_split[-1] == '' else None
        sentence_num = None

        for i, sentence in enumerate(context_split):
            if "[HERE]" in sentence:
                sentence_num = i
                break

        truncate = ['left', 'right']

        if sentence_num == 0:
            truncate_method = 'right'

        elif sentence_num == len(context_split) - 1:
            truncate_method = 'left'

        else:
            truncate_method = random.choice(truncate)

        if truncate_method == 'left':
            context_split_ = context_split[sentence_num-1:]

        else:
            context_split_ = context_split[:sentence_num+2]

        example['context'] = ''
        for elem in context_split_:
            example['context'] += elem + '다.'

        example['answers']['answer_start'][0] = example['context'].find("[HERE]")
        example['context'] = example['context'].replace("[HERE]", "")
        answer_start = example['answers']['answer_start'][0]

        # print(example['context'])
        # print(example['context'][answer_start:answer_start + len(example['answers']['text'][0])])

        if example['context'][answer_start:answer_start + len(example['answers']['text'][0])] != example['answers']['text'][0]:
            print(example['answers']['text'][0])
            print(example['context'][answer_start:answer_start + len(example['answers']['text'][0])])

        return example

    train_dataset_ = train_dataset_.map(
        preprocess,
        with_indices=True,
    )

    #print_sample(train_dataset_)

    return datasets.concatenate_datasets([train_dataset, train_dataset_]) if concat else datasets.concatenate_datasets([train_dataset_left, train_dataset_])

def AEDA(train_dataset : datasets.Dataset, tokenizer : dict, ratio=0.3, min_puncation=3, max_puncation=4, concat=True):
    random_idx = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio) if ratio <= 1 else ratio, replace=False)
    train_dataset_ = train_dataset.select(random_idx)

    # get left data that are not in choice
    left_choice = list(set(range(len(train_dataset))) - set(random_idx))
    train_dataset_left = train_dataset.select(left_choice)
    # print(train_dataset_[:1]['context'])

    def preprocess(example, id_start):
        punctuation_list = ['.', ',', '!', '?', ':', ';']
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
        preprocess,
        with_indices=True,
    )

    # print()
    # print(train_dataset_[:1]['context'])

    return datasets.concatenate_datasets([train_dataset, train_dataset_]) if concat else train_dataset_

def swap_sentence(train_dataset : datasets.Dataset, tokenizer : dict, ratio=0.33, concat=True):
    random_idx = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio) if ratio <= 0.33 else ratio, replace=False)
    train_dataset_ = train_dataset.select(random_idx)

    # get left data that are not in choice
    left_choice = list(set(range(len(train_dataset))) - set(random_idx))
    train_dataset_left = train_dataset.select(left_choice)

    # for k, v in train_dataset_[:1].items():
    #     print(f"{k} : {v}")
    # answer_start = train_dataset_[:1]['answers'][0]['answer_start'][0]
    # answer_end = answer_start + len(train_dataset_[:1]['answers'][0]['text'][0])
    # print(f"answer_start : {answer_start}, answer_end : {answer_end}")
    # print(f"answer : {train_dataset_[:1]['context'][0][answer_start:answer_end]}")

    def preprocess_random_truncation(example, id_start):
        id_start = 1000000

        # ID 및 인덱스 설정
        example['id'] = f"mrc-id-0-{id_start}"
        example['__index_level_0__'] = id_start

        original_answer_start = example['answers']['answer_start'][0]
        original_context = example['context']

        answer_start = example['answers']['answer_start'][0]

        context = example['context'][0:answer_start] + "[HERE]" + example['context'][answer_start:]
        answer_start += len("[HERE]")

        # print(context)
        # print(context[answer_start:answer_start + len(example['answers']['text'][0])])

        context_split_by_line = context.split('다.')

        if len(context_split_by_line) == 1:
            return example

        # find a sentence number that contains the [HERE] token
        sentence_num = 0
        for i, sentence in enumerate(context_split_by_line):
            if answer_start < len(sentence):
                sentence_num = i
                break

            else:
                answer_start -= len(sentence)

        # swap this sentence with left or right sentence
        # only swap right if sentence index is 0
        if sentence_num == 0:
            swap_direction = 'right'

        elif sentence_num == len(context_split_by_line) - 1:
            swap_direction = 'left'

        else:
            swap_direction = random.choice(['left', 'right'])

        if swap_direction == 'left':
            context_split_by_line[sentence_num], context_split_by_line[sentence_num - 1] = context_split_by_line[
                sentence_num - 1], context_split_by_line[sentence_num]

        elif swap_direction == 'right':
            context_split_by_line[sentence_num], context_split_by_line[sentence_num + 1] = context_split_by_line[
                sentence_num + 1], context_split_by_line[sentence_num]

        example['context'] = ''
        for elem in context_split_by_line:
            example['context'] += elem + '다.'

        # find index of [HERE] token
        answer_start = example['context'].find("[HERE]")
        example['context'] = example['context'].replace("[HERE]", "")
        example['answers']['answer_start'][0] = answer_start

        if example['context'][answer_start:answer_start + len(example['answers']['text'][0])] != example['answers']['text'][0]:
            example['context'] = original_context
            example['answers']['answer_start'][0] = original_answer_start

        return example

    train_dataset_ = train_dataset_.map(
        preprocess_random_truncation,
        with_indices=True,
    )

    # print()
    # for k, v in train_dataset_[:1].items():
    #     print(f"{k} : {v}")
    # answer_start = train_dataset_[:1]['answers'][0]['answer_start'][0]
    # answer_end = answer_start + len(train_dataset_[:1]['answers'][0]['text'][0])
    # print(f"answer_start : {answer_start}, answer_end : {answer_end}")
    # print(f"answer : {train_dataset_[:1]['context'][0][answer_start:answer_end]}")

    return datasets.concatenate_datasets([train_dataset, train_dataset_]) if concat else datasets.concatenate_datasets([train_dataset_left, train_dataset_])

def stop_word(train_dataset : datasets.Dataset, tokenizer : dict, ratio=1, concat=False):
    random_idx = np.random.choice(len(train_dataset), int(len(train_dataset) * ratio) if ratio <= 1 else ratio, replace=False)
    train_dataset_ = train_dataset.select(random_idx)

    # get left data that are not in choice
    left_choice = list(set(range(len(train_dataset))) - set(random_idx))
    train_dataset_left = train_dataset.select(left_choice)

    #print_sample(train_dataset_)
    #nltk.download('punkt_tab')

    def preprocess_random_truncation(example, id_start):
        id_start = 1000000

        # ID 및 인덱스 설정
        example['id'] = f"mrc-id-0-{id_start}"
        example['__index_level_0__'] = id_start

        original_answer_start = example['answers']['answer_start'][0]
        original_context = example['context']

        answer_start = example['answers']['answer_start'][0]
        context = example['context'][0:answer_start] + "HEREHERE" + example['context'][answer_start:]

        # each line of stopwords.txt is stopword elem
        stopwords = []

        with open(os.path.join(os.getcwd(), 'src', 'stopwords.txt'), 'r', encoding='utf-8') as f:
            for line in f:
                stopwords.append(line.strip())

        trim_words = word_tokenize(context)

        example['context'] = ''
        for word in trim_words:
            if word not in stopwords:
                example['context'] += word + ' '

        answer_start = example['context'].find("HEREHERE")
        example['answers']['answer_start'][0] = answer_start
        example['context'] = example['context'].replace("HEREHERE", "")

        return example

    train_dataset_ = train_dataset_.map(
        preprocess_random_truncation,
        with_indices=True,
    )

    #print_sample(train_dataset_)

    return datasets.concatenate_datasets([train_dataset, train_dataset_]) if concat else datasets.concatenate_datasets([train_dataset_left, train_dataset_])

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

# def sample_test(train_dataset : datasets.Dataset):
#     train_dataset_ = copy.deepcopy(train_dataset)
#     random_idx = int(np.random.choice(len(train_dataset), 1)[0])
#     answer_start = train_dataset_[random_idx]['answers']['answer_start'][0]
#
#     example = train_dataset_[random_idx]
#
#     # insert "[HERE]" token to answer_start
#     context = example['context'][0:answer_start] + "[HERE]" + example['context'][answer_start:]
#     answer_start += len("[HERE]")
#     print(context)
#     print(context[answer_start:answer_start + len(example['answers']['text'][0])])
#
#
#     # for k, v in sample.items():
#     #     print(f"{k} : {v}")
#     #     print()
#     context_split_by_line = context.split('.')
#
#     if len(context_split_by_line) == 1:
#         return example
#
#     # find a sentence number that contains the [HERE] token
#     sentence_num = 0
#     for i, sentence in enumerate(context_split_by_line):
#         if answer_start < len(sentence):
#             sentence_num = i
#             break
#
#         else:
#             answer_start -= len(sentence)
#
#     # swap this sentence with left or right sentence
#     swap_direction = random.choice(['left', 'right'])
#
#     # only swap right if sentence index is 0
#     if sentence_num == 0:
#         swap_direction = 'right'
#
#     elif sentence_num == len(context_split_by_line) - 1:
#         swap_direction = 'left'
#
#     else:
#         swap_direction = random.choice(['left', 'right'])
#
#     if swap_direction == 'left':
#         context_split_by_line[sentence_num], context_split_by_line[sentence_num - 1] = context_split_by_line[sentence_num - 1], context_split_by_line[sentence_num]
#
#     elif swap_direction == 'right':
#         context_split_by_line[sentence_num], context_split_by_line[sentence_num + 1] = context_split_by_line[sentence_num + 1], context_split_by_line[sentence_num]
#
#     example['context'] = ''
#     for elem in context_split_by_line:
#         example['context'] += elem + '.'
#
#     # find index of [HERE] token
#     answer_start = example['context'].find("[HERE]")
#     example['context'] = example['context'].replace("[HERE]", "")
#     example['answers']['answer_start'][0] = answer_start - len("[HERE]")
#
#     print(example['context'])
#     print(example['context'][answer_start:answer_start + len(example['answers']['text'][0])])
#
#     return example
def sample_test(train_dataset : datasets.Dataset):
    train_dataset_ = copy.deepcopy(train_dataset)
    random_idx = int(np.random.choice(len(train_dataset), 1)[0])

    context = train_dataset_[random_idx]['context']
    example = train_dataset_[random_idx]
    answer_start = train_dataset_[random_idx]['answers']['answer_start'][0]

    print(context)
    print(context[answer_start:answer_start + len(example['answers']['text'][0])])

    context = example['context'][0:answer_start] + "[HERE]" + example['context'][answer_start:]
    answer_start_ = answer_start + len("[HERE]")


    context_split = context.split('.')
    sentence_num = 0

    for i, sentence in enumerate(context_split):
        if answer_start_ < len(sentence):
            sentence_num = i
            break

        else:
            answer_start_ -= len(sentence)

    truncate = ['left', 'right']

    if sentence_num == 0:
        truncate_method = 'right'

    elif sentence_num == len(context_split) - 1:
        truncate_method = 'left'

    else:
        truncate_method = random.choice(truncate)

    if truncate_method == 'left':
        context_split = context_split[sentence_num-1:]

    elif truncate_method == 'right':
        context_split = context_split[:sentence_num+1]

    example['context'] = ''
    for elem in context_split:
        example['context'] += elem + '.'

    example['answers']['answer_start'][0] = example['context'].find("[HERE]")
    example['context'] = example['context'].replace("[HERE]", "")

    answer_start = example['answers']['answer_start'][0]
    print(example['context'])
    print(example['context'][answer_start:answer_start + len(example['answers']['text'][0])])

    return example

if __name__ == "__main__":
    train_dataset_dir = "../data/train_dataset/train/dataset.arrow"
    train_dataset = datasets.Dataset.from_file(train_dataset_dir)

    #train_dataset = augmentation(train_dataset, tokenizer={})
    #analysis()
    #swap_sentence(train_dataset, tokenizer={}, ratio=1)
    #random_truncation_all(train_dataset)
    stop_word(train_dataset, tokenizer={}, ratio=1)