import json
import os
import pickle
import time
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_from_disk
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from transformers import AutoTokenizer


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        args,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        """Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.
        """

        self.tokenizer = tokenize_fn
        self.data_path = data_path
        self.args = args
        
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        
        self.bm25 = None  # get_sparse_embedding()로 생성합니다

    def get_sparse_embedding(self) -> None:
        """Passage Embedding을 만들고 BM25를 초기화합니다."""
        pickle_name = "bm25_sparse_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            tokenized_corpus = [self.tokenizer(doc) for doc in self.contexts]
            self.bm25 = BM25Okapi(tokenized_corpus)
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.
        """

        assert self.bm25 is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
                
            total = []
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context": " ".join([self.contexts[pid] for pid in doc_indices[idx]]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """

        tokenized_query = [self.tokenizer(query)]
        result = np.array([self.bm25.get_scores(query) for query in tokenized_query])
        doc_scores = []
        doc_indices = []
        
        for scores in result:
            sorted_result = np.argsort(scores)[-k:][::-1]
            doc_scores.append(scores[sorted_result])
            doc_indices.append(sorted_result.tolist())
        
        return doc_scores, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """Arguments:
            queries (List):
                여러 개의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """

        tokenized_queries = [self.tokenizer(query) for query in queries]
        result = np.array([self.bm25.get_scores(query) for query in tokenized_queries])
        doc_scores = []
        doc_indices = []
        
        for scores in result:
            sorted_result = np.argsort(scores)[-k:][::-1]
            doc_scores.append(scores[sorted_result])
            doc_indices.append(sorted_result.tolist())
        
        return doc_scores, doc_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_name", metavar="./data/train_dataset", type=str, help="")
    parser.add_argument("--model_name_or_path", metavar="bert-base-multilingual-cased", type=str, help="")
    parser.add_argument("--data_path", metavar="./data", type=str, help="")
    parser.add_argument("--context_path", metavar="wikipedia_documents.json", type=str, help="")

    args = parser.parse_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )

    tokenizer = AutoTokenizer.from_pretrained("uomnf97/klue-roberta-finetuned-korquad-v2", use_fast=False)

    retriever = BM25SparseRetrieval(
        tokenize_fn=tokenizer.tokenize,
        args=args,
        data_path=args.data_path,
        context_path=args.context_path,
    )

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    # test single query
    with timer("single query by exhaustive search using bm25"):
        scores, indices = retriever.retrieve(query)

    # test bulk
    with timer("bulk query by exhaustive search using bm25"):
        df = retriever.retrieve(full_ds)
        df["correct"] = df["original_context"] == df["context"]
        print(
            "correct retrieval result by exhaustive search",
            df["correct"].sum() / len(df),
        )
