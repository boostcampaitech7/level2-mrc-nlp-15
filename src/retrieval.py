import datetime
import logging
import os
import sys
import numpy as np
from typing import Callable, List, NoReturn, Tuple

from arguments import DataTrainingArguments, ModelArguments
from datasets import (
    Dataset,
    DatasetDict,
    Features,
    Value,
    Sequence,
    load_from_disk,
    load_metric
)
from qa_trainer import QATrainer
from retrieval_BM25 import BM25SparseRetrieval
from retrieval_hybridsearch import HybridSearch
from retrieval_Dense import DenseRetrieval
from retrieval_2s_rerank import TwoStageReranker

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
)
from utils import set_seed, check_no_error, postprocess_qa_predictions


class Retriever:
    def __init__(
        self,
        tokenize_fn,
        args,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        name: str
     ):
        self.retriever = None
        if name == "2s_rerank":
            self.retriever = TwoStageReranker(
                tokenize_fn=tokenize_fn,
                args=data_args,  
                data_path=data_path,
                context_path=context_path
            )
        elif name == "BM25":
            self.retriever = BM25SparseRetrieval(
                tokenize_fn=tokenize_fn,
                # args=data_args,  
                data_path=data_path,
                context_path=context_path
            )
            self.retriever.get_sparse_embedding()
        elif name == "Dense":
            self.retriever = DenseRetrieval(
                data_path=data_path,
                context_path=context_path
            )
            self.retriever.get_dense_embedding()
        elif name == "hybridsearch":
            self.retriever = HybridSearch(
                tokenize_fn=tokenize_fn,
                # args=data_args,
                data_path=data_path,
                context_path=context_path
            )
            self.retriever.get_sparse_embedding()
            self.retriever.get_dense_embedding()
        elif name == "tfidf":
            self.retriever = TFIDFRetrieval(
                tokenize_fn=tokenize_fn,
                args=data_args, 
                data_path=data_path,
                context_path=context_path
            )
            self.retriever.get_sparse_embedding()

    def retrieve(self, query_or_dataset):
        return self.retriever.retrieve(query_or_dataset)