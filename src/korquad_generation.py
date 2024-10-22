import os.path
import re
import datasets
from bs4 import BeautifulSoup
import MeCab
import numpy as np

replace_list = [
    '이 문서는 미공개 또는 계획만 발표되었거나, 현재 진행 중인 작품의 내용을 포함하고 있습니다. 내용에 대한 의견이 있으시다면  토론 문서에서 나누어 주세요.정확한 내용을 반영할 수 있도록 문서 수정을 도와주세요.',
    '이 분야를 잘 알고 계신다면 이 문서가 더 좋은 문서가 되도록 문서 수정을 도와주세요.',
    '관련된 위키프로젝트인 위키프로젝트 법학에서 도움을 구하실 수도 있습니다.',
    '\n\n\n',
    '\n\n',
    '\n',
    '     ',
    '    ',
    '   ',
    '  ',
    '&nbsp;',
    '[]',
    ' ',
]
mapper = dict()
def preprocess(example, max_len):
    example.pop('url')
    example.pop('raw_html')
    example['answer'].pop('html_answer_start')

    # add [HERE] token to the answer text in context
    example['context'] = example['context'][:example['answer']['answer_start']] + "[HEREHERE]" + example['context'][example['answer']['answer_start']:]
    example['context'] = re.sub(r'<table.*?>.*?</table>', '', example['context'], flags=re.DOTALL)
    example['context'] = re.sub(r'</span><a>편집</a><span>', '', example['context'], flags=re.DOTALL)
    example['context'] = re.sub(r'\[\d+\]', '', example['context'], flags=re.DOTALL)

    start_index = example['context'].index("<a>검색하러 가기") + len("<a>검색하러 가기")
    end_index = [example['context'].find("<div>원본 주소"), example['context'].find("<h2><span></span><span>외부 링크"), example['context'].find("<h2><span></span><span>각주"), example['context'].find("<h2><span></span><span>참고 문헌"), example['context'].find("<h2><span></span><span>같이 보기")]

    # end index is min index that is not -1
    end_index = min([i for i in end_index if i != -1])

    example['context'] = example['context'][start_index:end_index]

    soup = BeautifulSoup(example['context'], 'html.parser')
    example['context'] = soup.get_text()

    soup = BeautifulSoup(example['answer']['text'], 'html.parser')
    example['answer']['text'] = soup.get_text()

    for elem in replace_list:
        example['context'] = example['context'].replace(elem, " ")
        example['answer']['text'] = example['answer']['text'].replace(elem, " ")

    example['context'] = example['context'].strip()
    token_idx = example['context'].find("[HEREHERE]")

    if token_idx == -1:
        example['id'] = "-1"
        return example

    if example['answer']['text'] not in example['context']:
        example['id'] = "-1"
        return example

    half_max = max_len // 2
    new_end_idx = min(token_idx + half_max, len(example['context']))
    new_start_idx = max(token_idx - half_max, 0)

    example['context'] = example['context'][new_start_idx:new_end_idx]

    if example['title'] not in example['context']:
        # append title in front of context
        example['context'] = example['title'] + "\n" + example['context']

    token_idx = example['context'].index("[HEREHERE]")

    example['answer']['answer_start'] = token_idx
    example['context'] = example['context'].replace("[HEREHERE]", '')

    prob = example['answer']['text']
    actual = example['context'][token_idx:token_idx + len(example['answer']['text'])]

    if prob != actual:
        example['id'] = "-1"
        return example

    return example

def post_process(example):
    if isinstance(example['answer']['text'], str):
        example['answer']['text'] = [example['answer']['text']]  # 리스트로 변환

    # answer['answer_start']가 정수라면 리스트로 변환
    if isinstance(example['answer']['answer_start'], int):
        example['answer']['answer_start'] = [example['answer']['answer_start']]  # 리스트로 변환

    example['answers'] = example.pop('answer')

    return example

def validation_(dataset):
    # print length of dataset
    print(f"Length of dataset train: {len(dataset['train'])}")
    print(f"Length of dataset validation: {len(dataset['validation'])}")

    example_t = dataset['train']
    example_v = dataset['validation']

    drop_list = []

    for i, each in enumerate(example_t):
        prob = each['answers']['text'][0]
        actual = each['context'][each['answers']['answer_start'][0]:each['answers']['answer_start'][0] + len(each['answers']['text'][0])]

        if prob != actual:
            print(f"DIFF! : prob\n{prob}\nactual\n{actual}\n")
            drop_list.append(i)

    t = example_t.select(
        indices=[i for i in range(len(example_t)) if i not in drop_list]
    )

    drop_list = []

    for i, each in enumerate(example_v):
        prob = each['answers']['text'][0]
        actual = each['context'][each['answers']['answer_start'][0]:each['answers']['answer_start'][0] + len(each['answers']['text'][0])]

        example_v[i]['answers']['text'] = list(example_t[i]['answers']['text'])
        example_v[i]['answers']['answer_start'] = list(example_t[i]['answers']['answer_start'])

        if prob != actual:
            print(f"DIFF! : prob\n{prob}\nactual\n{actual}\n")
            drop_list.append(i)

    v = example_v.select(
        indices=[i for i in range(len(example_v)) if i not in drop_list]
    )

    dataset['train'] = t
    dataset['validation'] = v

    print("trimmed len train : ", len(dataset['train']))
    print("trimmed len valid : ", len(dataset['validation']))

    return dataset

def run_4():
    kq_4 = datasets.load_dataset("squad_kor_v1")

    kq_4_train_choice = np.random.choice(len(kq_4['train']), int(len(kq_4['train']) * 0.33))
    kq_4['train'] = kq_4['train'].select(kq_4_train_choice)

    return kq_4

def renew_dataset(dataset):
    import pandas as pd
    # Function to filter unique titles
    def filter_unique_titles(dataset):
        df = pd.DataFrame(dataset)
        return df.to_dict('records')

    # Apply the filter to both train and validation splits
    train_filtered = filter_unique_titles(dataset['train'])
    validation_filtered = filter_unique_titles(dataset['validation'])

    # Create a new dataset with filtered data
    filtered_dataset = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(pd.DataFrame(train_filtered)),
        'validation': datasets.Dataset.from_pandas(pd.DataFrame(validation_filtered))
    })

    return filtered_dataset

def run_3():
    parent_dir = os.path.dirname(os.path.dirname(os.getcwd()))
    kq_3 = datasets.load_from_disk(os.path.join(parent_dir, "data", "train_dataset"))
    mecab = MeCab.Tagger()

    def preprocess(example):
        #this preprocess will add noun, pronoun at first of the context of the question
        #it will use mecab to get the noun, pronoun

        question = example['question']
        parsed = mecab.parse(question)

        nouns_pronouns = []
        for line in parsed.splitlines():
            if line == "EOS":
                break
            # 형태소와 품사 정보 분리
            word, info = line.split('\t')
            pos = info.split(',')[0]
            # 명사(NNG, NNP) 또는 대명사(NP) 추출
            if pos in ["NNG", "NNP", "NP"]:
                nouns_pronouns.append(word)

        example['question'] = ", ".join(nouns_pronouns) + " " + example['question']

        # Extract noun, pronoun from the question

        return example

    kq_3 = kq_3.map(
        preprocess,
        num_proc=1,
    )

    return kq_3

def run_2():
    kq_2 = datasets.load_dataset("squad_kor_v2")

    kq_2 = kq_2.map(
        preprocess,
        fn_kwargs={'max_len': 1024},
        num_proc=8,
    )

    kq_2 = kq_2.filter(lambda x: x['id'] != "-1" and len(x['answer']['text']) < 64)

    return kq_2

def run_1():
    kq_1 = datasets.load_dataset("squad_kor_v1")
    kq_1 = kq_1.shuffle(seed=42)

    import pandas as pd
    # Function to filter unique titles
    def filter_unique_titles(dataset):
        df = pd.DataFrame(dataset)
        unique_titles = df.drop_duplicates(subset=['title'])
        return unique_titles.to_dict('records')

    # Apply the filter to both train and validation splits
    train_filtered = filter_unique_titles(kq_1['train'])
    validation_filtered = filter_unique_titles(kq_1['validation'])

    # Create a new dataset with filtered data
    filtered_dataset = datasets.DatasetDict({
        'train': datasets.Dataset.from_pandas(pd.DataFrame(train_filtered)),
        'validation': datasets.Dataset.from_pandas(pd.DataFrame(validation_filtered))
    })

    mecab = MeCab.Tagger()
    def postprocess(example):
        #this preprocess will add noun, pronoun at first of the context of the question
        #it will use mecab to get the noun, pronoun

        question = example['question']
        parsed = mecab.parse(question)

        nouns_pronouns = []
        for line in parsed.splitlines():
            if line == "EOS":
                break
            # 형태소와 품사 정보 분리
            word, info = line.split('\t')
            pos = info.split(',')[0]
            # 명사(NNG, NNP) 또는 대명사(NP) 추출
            if pos in ["NNG", "NNP", "NP"]:
                nouns_pronouns.append(word)

        example['question'] = ", ".join(nouns_pronouns) + " " + example['question']

        # Extract noun, pronoun from the question

        return example

    filtered_dataset = filtered_dataset.map(
        postprocess,
        num_proc=1,
    )

    return filtered_dataset

if __name__ == '__main__':
    name = 'korquad_1_full'
    parent_dir = os.path.dirname(os.getcwd())
    # kq_2 = run_2()
    # kq_2.save_to_disk(os.path.join(os.getcwd(), name))
    # kq_2 = datasets.load_from_disk(os.path.join(os.getcwd(), name))
    #
    # kq_2 = kq_2.map(
    #     post_process,
    #     num_proc=8,
    # )
    #
    #
    # kq_2.save_to_disk(os.path.join(os.getcwd(), "korquad_2_full_1"))
    # kq_2['train'].to_csv(os.path.join(os.getcwd(), "korquad_2_full_1.json"), encoding='utf-16')

    # kq_1 = run_1()
    # kq_1 = validation_(kq_1)
    # kq_1.save_to_disk(os.path.join(os.getcwd(), "train_korquad1_Mecab"))
    #
    # # kq_2 = run_2()
    # # kq_2 = validation_(kq_2)
    # # kq_2.save_to_disk(os.path.join(os.getcwd(), "train_dataset_Mecab"))
    # #
    #
    # kq_3 = run_3()
    # kq_3 = validation_(kq_3)
    # kq_3.save_to_disk(os.path.join(os.getcwd(), "train_dataset_Mecab"))

    kq_4 = run_4()
    kq_4 = validation_(kq_4)
    kq_4 = renew_dataset(kq_4)

    kq_1 = datasets.load_from_disk(os.path.join(parent_dir, "data", "train_dataset"))

    kq_1['train'] = datasets.concatenate_datasets([kq_1['train'], kq_4['train']])
    kq_1['validation'] = datasets.concatenate_datasets([kq_1['validation'], kq_4['validation']])

    kq_1['train'] = kq_1['train'].filter(lambda x: len(x['answers']['text']) <= 64)
    kq_1['validation'] = kq_1['validation'].filter(lambda x: len(x['answers']['text']) <= 64)

    print(f"Length of dataset train: {len(kq_1['train'])}")
    print(f"Length of dataset validation: {len(kq_1['validation'])}")
    kq_1 = kq_1.shuffle(42)

    # divide kq_1 in three parts
    kq_1_1_t = kq_1['train'].select(indices=[i for i in range(len(kq_1['train'])) if i % 3 == 0])
    kq_1_2_t = kq_1['train'].select(indices=[i for i in range(len(kq_1['train'])) if i % 3 == 1])
    kq_1_3_t = kq_1['train'].select(indices=[i for i in range(len(kq_1['train'])) if i % 3 == 2])

    kq_1_1_v = kq_1['validation'].select(indices=[i for i in range(len(kq_1['validation'])) if i % 3 == 0])
    kq_1_2_v = kq_1['validation'].select(indices=[i for i in range(len(kq_1['validation'])) if i % 3 == 1])
    kq_1_3_v = kq_1['validation'].select(indices=[i for i in range(len(kq_1['validation'])) if i % 3 == 2])

    s1 = datasets.DatasetDict({
        'train': kq_1_1_t,
        'validation': kq_1_1_v
    })

    s2 = datasets.DatasetDict({
        'train': kq_1_2_t,
        'validation': kq_1_2_v
    })

    s3 = datasets.DatasetDict({
        'train': kq_1_3_t,
        'validation': kq_1_3_v
    })

    s1.save_to_disk(os.path.join(parent_dir, "data", "train_dataset_and_korquad_20000_1"))
    s2.save_to_disk(os.path.join(parent_dir, "data", "train_dataset_and_korquad_20000_2"))
    s3.save_to_disk(os.path.join(parent_dir, "data", "train_dataset_and_korquad_20000_3"))



    # concat kq_1 and kq_3
    # kq_1['train'] = datasets.concatenate_datasets([kq_1['train'], kq_3['train']])
    # kq_1['validation'] = datasets.concatenate_datasets([kq_1['validation'], kq_3['validation']])
    # kq_1 = kq_1.shuffle(42)
    # kq_1.save_to_disk(os.path.join(os.getcwd(), "train_dataset_and_korquad_Mecab"))