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
from sparsembed import model, retrieve
from transformers import AutoModelForMaskedLM, AutoTokenizer

from utils import set_seed

set_seed(42)
logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    logging.info(f"[{name}] done in {time.time() - t0:.3f} s")


class SpldRetrieval:
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
        self.df = None

        self.model = model.Splade(
            model=AutoModelForMaskedLM.from_pretrained("naver/splade_v2_max").to(device),
            tokenizer=AutoTokenizer.from_pretrained("naver/splade_v2_max"),
            device=device
        )
        self.retriever = retrieve.SpladeRetriever(
            key="id", 
            on=["text"], 
            model=self.model 
        )

    def get_sparse_embedding(self, df=None) -> NoReturn:
        if df is None:
            self.df = [{'id': i, 'text': c} for i, c in zip(self.ids, self.contexts)]
        else:
            self.df = df

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List[float], List[str]], pd.DataFrame]:
        assert self.df is not None, "You should first execute `get_sparse_embedding()`"

        # print(self.df.to_dict())
        self.retriever = self.retriever.add(
            documents=self.df,
            batch_size=10,
            k_tokens=256, 
        )

        result = self.retriever(
            query_or_dataset,
            k_tokens=20, # Maximum number of activated tokens.
            k=100, # Number of documents to retrieve.
            batch_size=10
        )

        doc_scores = [] 
        doc_indices = []
        for i in range(len(result)):
            doc_scores.append([dic['similarity'] for k, dic in enumerate(result[i]) if k < topk])
            doc_indices.append([dic['id'] for k, dic in enumerate(result[i]) if k < topk])
        for i in range(topk):
            logging.info(f"Top-{i+1} passage with score {doc_scores[0][i]:4f}")
            logging.info(self.contexts[doc_indices[0][i]])
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

    retriever = SpldRetrieval(
        data_path=args.data_path,
        context_path=args.context_path,
    )

    retriever.get_sparse_embedding()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    with timer("single query by exhaustive search"):
        doc_scores, doc_indices = retriever.retrieve(query, topk=1)