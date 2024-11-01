import json
import os
import pickle
import time
import torch
import logging
import scipy
import scipy.sparse
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union, NoReturn
from tqdm.auto import tqdm

import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from torch.nn.functional import normalize

from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi, BM25Plus
from transformers import AutoTokenizer, AutoModel
from utils import set_seed

set_seed(42)
logger = logging.getLogger(__name__)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logging.info(f"[{name}] done in {time.time() - t0:.3f} s")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class HybridSearch:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        corpus: Optional[pd.DataFrame] = None
    ) -> NoReturn:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        logging.info(f"Lengths of contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tokenize_fn=tokenize_fn

        self.dense_model_name= 'intfloat/multilingual-e5-large-instruct' #'sentence-transformers/paraphrase-multilingual-mpnet-base-v2' #'BM-K/KoSimCSE-roberta' #'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        self.dense_tokenize_fn = AutoTokenizer.from_pretrained(
            self.dense_model_name
        )
        # self.sparse_embeder = TfidfVectorizer(
        #     tokenizer=self.tokenize_fn, ngram_range=(1, 2), max_features=50000,
        # )
        self.spares_embeder = None
        self.dense_embeder = AutoModel.from_pretrained(
            self.dense_model_name
        )
        self.sparse_embeds = None
        self.dense_embeds = None

    def get_sparse_embedding(self, question=None):
        vectorizer_path = os.path.join(self.data_path, "BM25Plus_sparse_vectorizer.bin")
        embeddings_path = os.path.join(self.data_path, "BM25Plus_sparse_embedding.bin")
        # vectorizer_path = os.path.join(self.data_path, "sparse_vectorizer.bin")
        # embeddings_path = os.path.join(self.data_path, "sparse_embedding.bin")

        if question is None:
            if os.path.isfile(vectorizer_path) and os.path.isfile(embeddings_path):
                with open(vectorizer_path, "rb") as f:
                    self.sparse_embeder = pickle.load(f)
                with open(embeddings_path, "rb") as f:
                    self.sparse_embeds = pickle.load(f)
                print("Sparse vectorizer and embeddings loaded.")
            else:
                print("Fitting sparse vectorizer and building embeddings.")
                self.sparse_embeder = BM25Plus([self.tokenize_fn(doc) for doc in self.contexts], k1=1.837782128608009, b=0.587622663072072, delta=1.1490)
                # self.sparse_embeds = self.sparse_embeder.fit_transform(self.contexts)
                with open(vectorizer_path, "wb") as f:
                    pickle.dump(self.sparse_embeder, f)
                if self.dense_embeds is not None:
                    with open(embeddings_path, "wb") as f:
                        pickle.dump(self.sparse_embeds, f)
                print("Sparse vectorizer and embeddings saved.")
        else:
            # self.sparse_embeder가 CountVectorizer, TfidfVectorizer 등 객체 일 때에만 이 부분 사용
            if not hasattr(self.sparse_embeder, 'vocabulary_'):
                vectorizer_path = os.path.join(self.data_path, "sparse_vectorizer.bin")
                if os.path.isfile(vectorizer_path):
                    with open(vectorizer_path, "rb") as f:
                        self.sparse_embeder = pickle.load(f)
                    print("Sparse vectorizer loaded for transforming the query.")
                else:
                    raise ValueError("The Sparse vectorizer is not fitted. Please run get_sparse_embedding() first.")
            return self.sparse_embeder.transform(question)


    def get_dense_embedding(self, question=None):
        if question is None:
            model_n = self.dense_model_name.split('/')[1]
            pickle_name = f"{model_n}_dense_embedding.bin"
            emd_path = os.path.join(self.data_path, pickle_name)

            if os.path.isfile(emd_path):
                self.dense_embeds = torch.load(emd_path)
                print("Dense embedding loaded.")
            else:
                print("Building passage dense embeddings in batches.")
                self.dense_embeds = []
                batch_size = 64
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.dense_embeder.to(device)

                for i in tqdm(range(0, len(self.contexts), batch_size), desc="Encoding passages"):
                    batch_contexts = self.contexts[i:i+batch_size]
                    encoded_input = self.dense_tokenize_fn(
                        batch_contexts, padding=True, truncation=True, return_tensors='pt'
                    ).to(device)
                    with torch.no_grad():
                        model_output = self.dense_embeder(**encoded_input)
                    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                    sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
                    self.dense_embeds.append(sentence_embeddings.cpu())
                    del encoded_input, model_output, sentence_embeddings 
                    torch.cuda.empty_cache()  

                self.dense_embeds = torch.cat(self.dense_embeds, dim=0)
                torch.save(self.dense_embeds, emd_path)
                print("Dense embeddings saved.")
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dense_embeder.to(device)
            encoded_input = self.dense_tokenize_fn(
                question, padding=True, truncation=True, return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                model_output = self.dense_embeder(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)
            return sentence_embeddings.cpu()

    def hybrid_scale(self, dense_score, sparse_score, alpha: float):
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        if isinstance(dense_score, torch.Tensor):
            dense_score = dense_score.detach().numpy() 
        if isinstance(sparse_score, torch.Tensor):
            sparse_score = sparse_score.detach().numpy()  

        def min_max_normalize(score):
            return (score - np.min(score)) / (np.max(score) - np.min(score))
        def z_score_normalize(score):
            return (score - np.mean(score)) / np.std(score)

        # dense_score_normalized = min_max_normalize(dense_score)
        # sparse_score_normalized = min_max_normalize(sparse_score)
        # dense_score_normalized = z_score_normalize(dense_score)
        # sparse_score_normalized = z_score_normalize(sparse_score)

        # result = (1 - alpha) * dense_score_normalized + alpha * sparse_score_normalized
        result = (1 - alpha) * dense_score + alpha * sparse_score
        return result

    def get_similarity_score(self, q_vec, c_vec):
        # if isinstance(q_vec, scipy.sparse.spmatrix):
        #     q_vec = q_vec.toarray()  
        # if isinstance(c_vec, scipy.sparse.spmatrix):
        #     c_vec = c_vec.toarray()

        # q_vec = torch.tensor(q_vec)
        # c_vec = torch.tensor(c_vec)
        # return q_vec.matmul(c_vec.T)
   
        if isinstance(q_vec, scipy.sparse.spmatrix):
            q_vec = q_vec.toarray()  
        if isinstance(c_vec, scipy.sparse.spmatrix):
            c_vec = c_vec.toarray()
        
        q_vec = torch.tensor(q_vec, dtype=torch.float32)
        c_vec = torch.tensor(c_vec, dtype=torch.float32)

        if q_vec.ndim == 1:
            q_vec = q_vec.unsqueeze(0) 
        if c_vec.ndim == 1:
            c_vec = c_vec.unsqueeze(0) 

        similarity_score = torch.matmul(q_vec, c_vec.T)
        
        return similarity_score  

    def get_cosine_score(self, q_vec, c_vec):
        q_vec = q_vec / q_vec.norm(dim=1, keepdim=True)
        c_vec = c_vec / c_vec.norm(dim=1, keepdim=True)
        return torch.mm(q_vec, c_vec.T)

    def retrieve(self, query_or_dataset, topk: Optional[int] = 1, alpha: Optional[float] = 0.7):
        # assert self.sparse_embeds is not None, "You should first execute `get_sparse_embedding()`"
        assert self.dense_embeds is not None, "You should first execute `get_dense_embedding()`"

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, alpha, k=topk)
            logging.info(f"[Search query] {query_or_dataset}")

            for i in range(topk):
                logging.info(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                logging.info(self.contexts[doc_indices[i]])

            # return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)], doc_indices) # 임시로 doc_indices 반환 값으로 추가해둔 상태임 나중에 삭제할 것
            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], alpha, k=topk
                )
            for idx, example in enumerate(tqdm(query_or_dataset, desc="[Hybrid retrieval] ")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, alpha: float, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            # sparse_qvec = self.get_sparse_embedding([query])
            dense_qvec = self.get_dense_embedding([query])
        # assert sparse_qvec.nnz != 0, "Error: query contains no words in vocab."

        with timer("query ex search"):
            tokenized_query = [self.tokenize_fn(query)]
            sparse_score = np.array([self.sparse_embeder.get_scores(query) for query in tokenized_query])
            # sparse_score = self.get_similarity_score(sparse_qvec, self.sparse_embeds)
            # dense_score = self.get_cosine_score(dense_qvec, self.dense_embeds)
            dense_score = self.get_similarity_score(dense_qvec, self.dense_embeds)
            result = self.hybrid_scale(dense_score.numpy(), sparse_score, alpha)
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List[str], alpha: float, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        # sparse_qvec = self.get_sparse_embedding(queries)
        dense_qvec = self.get_dense_embedding(queries)
        # assert sparse_qvec.nnz != 0, "Error: query contains no words in vocab."
        
        tokenized_queries = [self.tokenize_fn(query) for query in queries]
        sparse_score = np.array([self.sparse_embeder.get_scores(query) for query in tokenized_queries])
        # sparse_score = self.get_similarity_score(sparse_qvec, self.sparse_embeds)
        # dense_score = self.get_cosine_score(dense_qvec, self.dense_embeds)
        dense_score = self.get_similarity_score(dense_qvec, self.dense_embeds)
        result = self.hybrid_scale(dense_score.numpy(), sparse_score, alpha)
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", default="../data/train_dataset", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-multilingual-cased", type=str)
    parser.add_argument("--data_path", default="../data", type=str)
    parser.add_argument("--context_path", default="wikipedia_documents.json", type=str)
    parser.add_argument("--use_faiss", default=False, type=bool)

    args = parser.parse_args()
    logging.info(args.__dict__)

    org_dataset = load_from_disk(args.dataset_name)
    if 'train' in org_dataset and 'validation' in org_dataset:
        full_ds = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
        )
    else:
        full_ds = org_dataset
    logging.info("*" * 40 + " query dataset " + "*" * 40)
    logging.info(f"Full dataset: {full_ds}")

    tokenizer = AutoTokenizer.from_pretrained("HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned")

    retriever = HybridSearch(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
    )
    retriever.get_dense_embedding()
    retriever.get_sparse_embedding()

    # query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    query = "유령은 어느 행성에서 지구로 왔는가?"

    with timer("single query by exhaustive search using hybrid search"):
        scores, contexts = retriever.retrieve(query, topk=20, alpha=0.0060115995634538455)
   
    for i, context in enumerate(contexts):
        print(f"Top-{i} 의 문서입니다. ")
        print("---------------------------------------------")
        print(context)


