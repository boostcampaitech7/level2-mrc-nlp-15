import json
import os
import pickle
import time
import torch
import logging
import scipy
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union, NoReturn
from tqdm.auto import tqdm

import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from torch.nn.functional import normalize

from retrieval_BM25 import BM25SparseRetrieval
from retrieval_Dense import DenseRetrieval

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


class TwoStageReranker:
    def __init__(
        self,
        tokenize_fn,
        # args,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> NoReturn:
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        logging.info(f"Lengths of contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.sparse_embeder = BM25SparseRetrieval(
            tokenize_fn=tokenize_fn,
            # args=args,
            data_path=data_path,
            context_path=context_path
        )
        self.dense_embeder = DenseRetrieval(
            data_path=data_path,
            context_path=context_path
        )
        self.sparse_embeds_bool = False
        self.dense_embeds_bool = False

    def retrieve_first(self, queries, topk: Optional[int] = 1):
        if self.sparse_embeds_bool == False:
            self.sparse_embeder.get_sparse_embedding()
            self.sparse_embeds_bool = True
        f_df = self.sparse_embeder.retrieve(queries, topk=topk)
        return f_df
    
    def retireve_second(self, queries, topk: Optional[int] = 1, contexts=None):
        self.dense_embeder.get_dense_embedding(contexts=contexts)
        s_df = self.dense_embeder.retrieve(queries, topk=topk)
        # self.sparse_embeder.get_sparse_embedding(contexts=contexts)
        # s_df = self.sparse_embeder.retrieve(queries, topk=topk)
        return s_df

    def retrieve(self, query_or_dataset, topk: Optional[int] = 1):
        retrieved_contexts = []
        if isinstance(query_or_dataset, str):
            _, doc_indices = self.retrieve_first(query_or_dataset, topk)
            retrieved_contexts = doc_indices
        elif isinstance(query_or_dataset, Dataset):
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                _, doc_indices = self.retrieve_first(example['question'], topk)
                retrieved_contexts.append(doc_indices)

        half_topk = int(topk / 3)

        if isinstance(query_or_dataset, str):
            second_df = self.retireve_second(query_or_dataset, half_topk, contexts=retrieved_contexts)
            return second_df
        elif isinstance(query_or_dataset, Dataset):
            second_df = []
            for i, example in enumerate(query_or_dataset):
                context = retrieved_contexts[i]
                doc_scores, doc_indices = self.retireve_second(example['question'], half_topk, contexts=context)
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join(doc_indices),
                }
                second_df.append(tmp)
            second_df = pd.DataFrame(second_df)
            return second_df

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

    tokenizer = AutoTokenizer.from_pretrained("HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned")

    # query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    query = "유령은 어느 행성에서 지구로 왔는가?"

    retriever = TwoStageReranker(
        tokenize_fn=tokenizer.tokenize,
        args=args,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    with timer("single query by exhaustive search"):
        doc_scores, doc_indices = retriever.retrieve(query, topk=5)