from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    # monologg/koelectra-small-v2-distilled-korquad-384
    # CurtisJeon/klue-roberta-large-korquad_v1_qa
    # uomnf97/klue-roberta-finetuned-korquad-v2
    model_name_or_path: str = field(
        default="uomnf97/klue-roberta-finetuned-korquad-v2",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    config_name_dpr: Optional[str] = field(
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
    #################################################################################
    batch_size: int = field(
        default=16
    )
    
    num_epochs: int = field(
        default=3
    )

    #################################################################################
    b_is_circle: bool = field(
        default=False
    )
    circle_dir: str = field(
        default="data/curriculum_dataset"
    )
    korquad: bool = field(
        default=False,
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default="data/train_dataset",
        metadata={"help": "The name of the dataset to use."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    max_answer_length: int = field(
        default=64,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    eval_retrieval: bool = field(
        default=True,
        metadata={"help": "Whether to run passage retrieval using sparse embedding."},
    )
    num_clusters: int = field(
        default=64, metadata={"help": "Define how many clusters to use for faiss."}
    )
    top_k_retrieval: int = field(
        default=40,
        metadata={
            "help": "Define how many top-k passages to retrieve based on similarity."
        },
    )
    use_faiss: bool = field(
        default=True, metadata={"help": "Whether to build with faiss"}
    )
    # save_total_limit: int = field(
    #     default = 1,
    #     metadata = {"help" : "Number of save model checkpoint"}
    # )
    dense_encoder_type: str = field(
        default = 'hybrid', metadata = {"help": "Whether to run passage retrieval using dense embedding."}
    )
    
    remove_char: bool = field(
        default=True, metadata={"help": "Whether to remove special character before embedding"}
    )
    data_path: str = field(
        default="data/",
        metadata={"help": "The path of the data directory"},
    )
    context_path: str = field(
        default="wikipedia_documents.json",
        metadata={"help": "The name of the context file"},
    )
