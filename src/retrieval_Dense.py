import json
import os
import time
import logging
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
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
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class DenseRetrieval:
    def __init__(
        self,
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

        self.tokenize_fn = AutoTokenizer.from_pretrained(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )
        self.dense_embeder = AutoModel.from_pretrained(
            'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
        )
        self.dense_embeds = None

    def get_dense_embedding(self, question=None, contexts=None):
        if contexts is not None:
            self.contexts = contexts
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dense_embeder.to(device)
            encoded_input = self.tokenize_fn(
                self.contexts, padding=True, truncation=True, return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                model_output = self.dense_embeder(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            self.dense_embeds = sentence_embeddings.cpu()

        if question is None and contexts is None:
            pickle_name = "dense_without_normalize_embedding.bin"
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
                    encoded_input = self.tokenize_fn(
                        batch_contexts, padding=True, truncation=True, return_tensors='pt'
                    ).to(device)
                    with torch.no_grad():
                        model_output = self.dense_embeder(**encoded_input)
                    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                    self.dense_embeds.append(sentence_embeddings.cpu())
                    del encoded_input, model_output, sentence_embeddings 
                    torch.cuda.empty_cache()  

                self.dense_embeds = torch.cat(self.dense_embeds, dim=0)
                torch.save(self.dense_embeds, emd_path)
                print("Dense embeddings saved.")
        elif question is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.dense_embeder.to(device)
            encoded_input = self.tokenize_fn(
                question, padding=True, truncation=True, return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                model_output = self.dense_embeder(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            return sentence_embeddings.cpu()

    def get_similarity_score(self, q_vec, c_vec):
        if isinstance(q_vec, scipy.sparse.spmatrix):
            q_vec = q_vec.toarray()  
        if isinstance(c_vec, scipy.sparse.spmatrix):
            c_vec = c_vec.toarray()

        q_vec = torch.tensor(q_vec)
        c_vec = torch.tensor(c_vec)
        return q_vec.matmul(c_vec.T).numpy()

    def get_cosine_score(self, q_vec, c_vec):
        q_vec = q_vec / q_vec.norm(dim=1, keepdim=True)
        c_vec = c_vec / c_vec.norm(dim=1, keepdim=True)
        return torch.mm(q_vec, c_vec.T).numpy()

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List[float], List[str]], pd.DataFrame]:
        assert self.dense_embeds is not None, "You should first execute `get_sparse_embedding()`"

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            logging.info(f"[Search query] {query_or_dataset}")

            for i in range(topk):
                logging.info(f"Top-{i+1} passage with score {doc_scores[i]:.6f}")
                logging.info(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(tqdm(query_or_dataset, desc="[Sparse retrieval] ")):
                retrieved_contexts = [self.contexts[pid] for pid in doc_indices[idx]]
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(retrieved_contexts),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            dense_qvec = self.get_dense_embedding(question=[query])

        with timer("query ex search"):
            result = self.get_cosine_score(dense_qvec, self.dense_embeds)
            # result = self.get_similarity_score(dense_qvec, self.dense_embeds)
        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List[str], k: Optional[int] = 1
    ) -> Tuple[List, List]:
        dense_qvec = self.get_dense_embedding(question=queries)
        
        result = self.get_cosine_score(dense_qvec, self.dense_embeds)
        # result = self.get_similarity_score(dense_qvec, self.dense_embeds)
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", default="../data/train_dataset", type=str)
    parser.add_argument("--data_path", default="../data", type=str)
    parser.add_argument("--context_path", default="wikipedia_documents.json", type=str)

    args = parser.parse_args()
    logging.info(args.__dict__)

    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )
    logging.info("*" * 40 + " query dataset " + "*" * 40)
    logging.info(f"Full dataset: {full_ds}")

    retriever = DenseRetrieval(
        data_path=args.data_path,
        context_path=args.context_path,
    )

    retriever.get_dense_embedding()

    # query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    query = "유령은 어느 행성에서 지구로 왔는가?"
    
    with timer("single query by exhaustive search"):
        scores, contexts = retriever.retrieve(query, topk=5)

    with timer("bulk query by exhaustive search"):
        df = retriever.retrieve(full_ds, topk=1)
        if "original_context" in df.columns:
            df["correct"] = df["original_context"] == df["context"]
            logging.info(f'correct retrieval result by exhaustive search: {df["correct"].sum() / len(df)}')