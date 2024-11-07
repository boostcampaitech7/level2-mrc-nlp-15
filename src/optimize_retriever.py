from rank_bm25 import BM25Plus
import optuna
from sklearn.metrics import ndcg_score
import numpy as np
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import normalize
import torch
from tqdm import tqdm

import logging
import datetime
import wandb

from retrieval_hybridsearch import HybridSearch  # HybridSearch 클래스가 정의된 파일에서 임포트
from retrieval_2s_rerank import TwoStageReranker

# 로그 설정
logger = logging.getLogger(__name__)
wandb.init(project="odqa",
           name="run_" + (datetime.datetime.now() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H%M%S"),
           entity="nlp15"
           )

datasets = load_from_disk("../data/train_dataset")

documents = datasets['train']['context']
queries = datasets['train']['question']

tokenizer = AutoTokenizer.from_pretrained("HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned")
dense_tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large-instruct')
dense_embeder = AutoModel.from_pretrained(
            'intfloat/multilingual-e5-large-instruct'
        )

dense_embeds = []
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dense_embeder.to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

for i in tqdm(range(0, len(documents), batch_size), desc="Encoding passages"):
    batch_contexts = documents[i:i+batch_size]
    encoded_input = dense_tokenizer(
        batch_contexts, padding=True, truncation=True, return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        model_output = dense_embeder(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
    dense_embeds.append(sentence_embeddings.cpu())
    del encoded_input, model_output, sentence_embeddings 
    torch.cuda.empty_cache()  

dense_embeds = torch.cat(dense_embeds, dim=0)

retriever = HybridSearch(
    tokenize_fn=tokenizer.tokenize,
    data_path="../data",
    context_path="wikipedia_documents.json"
)
# retriever = TwoStageReranker(
#     tokenize_fn=tokenizer.tokenize,
#     data_path="../data",
#     context_path="wikipedia_documents.json"
# )

retriever.get_dense_embedding()
retriever.get_sparse_embedding()

true_relevance_scores = np.eye(len(documents), dtype=int).tolist()

retriever.dense_embeds = dense_embeds
# retriever.dense_embeder.dense_embeds = dense_embeds

def objective(trial):
    alpha = trial.suggest_float("alpha", 0.4, 1.0)
    k1 = trial.suggest_float("k1", 0.5, 2.0)
    b = trial.suggest_float("b", 0.0, 1.0)
    delta = trial.suggest_float("delta", 0.0, 1.0)

    retriever.sparse_embeder = BM25Plus([tokenizer.tokenize(doc) for doc in documents], k1=k1, b=b, delta=delta)

    all_scores = []
    all_doc_indices = []

    for idx, query in enumerate(queries):
        scores, contexts, doc_indices = retriever.retrieve(query, topk=20, alpha=alpha)
        all_scores.append(scores)
        all_doc_indices.append(doc_indices)

    true_relevance_scores = []
    for idx, doc_indices in enumerate(all_doc_indices):
        relevance = [1 if doc_idx == idx else 0 for doc_idx in doc_indices]
        true_relevance_scores.append(relevance)

    all_scores = np.array(all_scores)
    true_relevance_scores = np.array(true_relevance_scores)

    avg_ndcg = ndcg_score(true_relevance_scores, all_scores)

    return avg_ndcg

class TQDMProgressBar:
    def __init__(self, n_trials):
        self.pbar = tqdm(total=n_trials)

    def __call__(self, study, trial):
        self.pbar.update(1)

n_trials = 30  # 총 시도 횟수
progress_bar = TQDMProgressBar(n_trials)
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=n_trials, callbacks=[progress_bar])

# 최적의 하이퍼파라미터 출력
print("Best alpha:", study.best_params["alpha"])
print("Best k1:", study.best_params["k1"])
print("Best b:", study.best_params["b"])
print("Best delta:", study.best_params["delta"])