import pandas as pd

from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    Sequence,
    load_from_disk,
    load_metric
)
from transformers import AutoTokenizer

from mecab import MeCab
from sklearn.feature_extraction.text import TfidfVectorizer

data_path = "/data/ephemeral/home/level2-mrc-nlp-15/data/train_dataset"
datasets = load_from_disk(data_path)
print(datasets)

train_datasets = pd.DataFrame(datasets['train'])
val_datasets = pd.DataFrame(datasets['validation'])

# val_datasets.head(20)
# for i , data in val_datasets.iterrows():
#     if i == 5: break
#     print(data['context'])
#     print(data['question'])
#     print("-----------------")

# model_path = 'klue/bert-base'
# tokenizer = AutoTokenizer.from_pretrained(model_path)

with open('/data/ephemeral/home/level2-mrc-nlp-15/data/stopwords.txt', encoding='utf-8') as f:
    stop_words = f.read().splitlines()

tokenizer = MeCab()

def filter_stop_words(tokenized_corpus, stop_words: list) -> list:
    return [x for x in tokenized_corpus if x not in stop_words]

tokenize_fn = lambda x: filter_stop_words(tokenizer.morphs(x), stop_words)

# tokenized_context = train_datasets['context'].map(lambda x: tokenizer.morphs(x))
# filtered_tokenized_context = []
# for context in tokenized_context:
#     filtered_tokenized_context.append([x for x in context if x not in stop_words])
# print(filtered_tokenized_context[123])
 
# print("-------------------------------------")

# tokenized_q = train_datasets['question'].map(lambda x: tokenizer.morphs(x))
# filtered_tokenized_q = []
# for q in tokenized_q:
#     filtered_tokenized_q.append([x for x in q if x not in stop_words])
# print(filtered_tokenized_q[123])

# 필터링된 코퍼스에서 각 context에 대해서 tf-idf 를 통해서 키워드 추출 후 해당 질문에 키워드가 존재하는지 확인하고 최종 비율 도출

tfidf = TfidfVectorizer(
    tokenizer=tokenize_fn,
    token_pattern=None
)

tfidf_matrix = tfidf.fit_transform(train_datasets['context'])
feature_names = tfidf.get_feature_names_out()

top_tokens_per_doc = []

for doc_idx, doc in enumerate(tfidf_matrix):
    # Sparse 벡터를 밀집 배열로 변환
    doc_array = doc.toarray().flatten()
    # TF-IDF 값이 큰 순서대로 상위 5개 인덱스 추출
    top_n = 5
    if len(doc_array) < top_n:
        top_n = len(doc_array)
    top_n_idx = doc_array.argsort()[-top_n:][::-1]
    # 상위 토큰과 해당 TF-IDF 값 추출
    top_tokens = [feature_names[i] for i in top_n_idx]
    top_tfidf_scores = [doc_array[i] for i in top_n_idx]
    # 결과 저장
    top_tokens_per_doc.append(top_tokens)
    # 출력
    # print(f"문서 {doc_idx}:")
    # for token, score in zip(top_tokens, top_tfidf_scores):
    #     print(f"단어: {token}, TF-IDF 값: {score}")

tokenized_q = train_datasets['question'].map(lambda x: tokenize_fn(x))

cnt = 0
for idx, q in enumerate(tokenized_q):
    ansdoc_tokens = top_tokens_per_doc[idx]

    val = 0
    for token in ansdoc_tokens:
        if token in q:
            val += 1

    if val > 0: cnt += 1

print(len(top_tokens_per_doc))
print(cnt)

