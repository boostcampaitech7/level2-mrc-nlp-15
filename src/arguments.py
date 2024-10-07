from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class ModelArguments:
    """
    Arguments related to the choice of model, configuration, and tokenizer for fine-tuning.
    """
    model_name_or_path: str = field(
        default="klue/bert-base",
        metadata={
            "help": "Path to the pretrained model or identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to data input for model training and evaluation.
    """
    dataset_name: Optional[str] = field(
        default="../data/train_dataset",
        metadata={
            "help": "The name of the dataset to use."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={
            "help": "Overwrite the cached training and evaluation sets"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of processes to use for preprocessing."
        },
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "Maximum total input sequence length after tokenization. Longer sequences are truncated, shorter ones are padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Pad all samples to `max_seq_length`. If False, dynamically pad samples during batching to the max length in the batch."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "Stride length when splitting long documents into chunks."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "Maximum length of an answer that can be generated."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={
            "help": "Whether to run passage retrieval using sparse embedding."
        },
    )
    num_clusters: int = field(
        default=64,
        metadata={
            "help": "Number of clusters to use for faiss indexing."
        },
    )
    top_k_retrieval: int = field(
        default=10,
        metadata={
            "help": "Number of top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=True,
        metadata={
            "help": "Enable faiss for efficient similarity search."
        },
    )
    retrieval_candidates: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Optional list of candidate passages for retrieval testing.",
        },
    )