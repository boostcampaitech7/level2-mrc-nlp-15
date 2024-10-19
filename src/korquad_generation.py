import os.path
import re
import datasets
from bs4 import BeautifulSoup

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

def run():
    kq_2 = datasets.load_dataset("squad_kor_v2")

    # debugging = np.random.choice(len(kq_2['validation']), 10, replace=False)
    # kq_2['train'] = kq_2['train'].select(debugging)
    # kq_2['validation'] = kq_2['validation'].select(debugging)

    kq_2 = kq_2.map(
        preprocess,
        fn_kwargs={'max_len': 1024},
        num_proc=8,
    )

    kq_2 = kq_2.filter(lambda x: x['id'] != "-1" and len(x['answer']['text']) < 64)

    return kq_2

if __name__ == '__main__':
    #torch.set_num_threads(1)

    name = 'korquad_2_full'

    kq_2 = run()
    kq_2.save_to_disk(os.path.join(os.getcwd(), name))
    kq_2 = datasets.load_from_disk(os.path.join(os.getcwd(), name))

    kq_2 = kq_2.map(
        post_process,
        num_proc=8,
    )

    kq_2 = validation_(kq_2)
    kq_2.save_to_disk(os.path.join(os.getcwd(), "korquad_2_full_1"))
    kq_2['train'].to_csv(os.path.join(os.getcwd(), "korquad_2_full_1.json"), encoding='utf-16')