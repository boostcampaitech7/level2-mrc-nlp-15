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
    #AutoModelForQuestionAnswering
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
)
from utils import set_seed, check_no_error, postprocess_qa_predictions
import wandb
from CNN_layer_model import CNN_RobertaForQuestionAnswering

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.save_steps = 0
    training_args.logging_steps = 10

    project_prefix = "[train]" if training_args.do_train else "[eval]" if training_args.do_eval else "[pred]"
    wandb.init(
        project="odqa",
        entity="nlp15",
        name=f"{project_prefix} {model_args.model_name_or_path.split('/')[0]}_{(datetime.datetime.now() + datetime.timedelta(hours=9)).strftime('%Y%m%d_%H%M%S')}",
        save_code=True,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"model is from {model_args.model_name_or_path}")
    logging.info(f"data is from {data_args.dataset_name}")

    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)
    print(">>> seed:", training_args.seed)


    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_fast=True,
    )
    #AutoModelForQuestionAnswering -> CNN_RobertaForQuestionAnswering : CNNlayer추가하여 성능향상
    model = CNN_RobertaForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    if training_args.do_predict and data_args.eval_retrieval:
        datasets = run_sparse_retrieval(
            tokenizer.tokenize, datasets, training_args, data_args,
        )

    run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def prepare_train_features(examples, tokenizer, question_column_name, pad_on_right, context_column_name, max_seq_length, data_args, answer_column_name):
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False, # True if bert, False if roberta
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = tokenized_examples.sequence_ids(i)

        sample_index = sample_mapping[i]
        answers = examples[answer_column_name][sample_index]

        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            if not (
                offsets[token_start_index][0] <= start_char
                and offsets[token_end_index][1] >= end_char
            ):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                while (
                    token_start_index < len(offsets)
                    and offsets[token_start_index][0] <= start_char
                ):
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def run_sparse_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    datasets: DatasetDict,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "../data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:

    retriever = HybridSearch(
        tokenize_fn=tokenize_fn,
        # args=data_args,  # args를 전달
        data_path=data_path,
        context_path=context_path
    )
    retriever.get_sparse_embedding()
    retriever.get_dense_embedding()

    # retriever = BM25SparseRetrieval(
    #     tokenize_fn=tokenize_fn,
    #     data_path=data_path,
    #     context_path=context_path
    # )
    # retriever.get_sparse_embedding()

    # retriever = TwoStageReranker(
    #     tokenize_fn=tokenize_fn,
    #     args=data_args,  # args를 전달
    #     data_path=data_path,
    #     context_path=context_path
    # )

    # retriever = DenseRetrieval(
    #     data_path=data_path,
    #     context_path=context_path
    # )
    # retriever.get_dense_embedding()

    # if data_args.use_faiss:
    #     retriever.build_faiss(num_clusters=data_args.num_clusters)
    #     df = retriever.retrieve_faiss(
    #         datasets["validation"], topk=data_args.top_k_retrieval
    #     )
    # else:

    # df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)
    # df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval, alpha=data_args.alpha_retrieval)
    df = retriever.retrieve(datasets["validation"], topk=10, alpha=0.7)


    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def prepare_validation_features(examples, tokenizer, question_column_name, pad_on_right, context_column_name, max_seq_length, data_args, answer_column_name):
    tokenized_examples = tokenizer(
        examples[question_column_name if pad_on_right else context_column_name],
        examples[context_column_name if pad_on_right else question_column_name],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_seq_length,
        stride=data_args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        return_token_type_ids=False, # True if bert, False if roberta
        padding="max_length" if data_args.pad_to_max_length else False,
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]
    return tokenized_examples


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> NoReturn:
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    pad_on_right = tokenizer.padding_side == "right"

    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        train_dataset = train_dataset.map(
            prepare_train_features,
            fn_kwargs={
                'tokenizer': tokenizer,
                'pad_on_right': pad_on_right,
                'max_seq_length': max_seq_length,
                'data_args': data_args,
                'question_column_name': question_column_name,
                'context_column_name': context_column_name,
                'answer_column_name': answer_column_name
            },
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval or training_args.do_predict:
        eval_dataset = datasets["validation"]

        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            fn_kwargs={
                'tokenizer': tokenizer,
                'pad_on_right': pad_on_right,
                'max_seq_length': max_seq_length,
                'data_args': data_args,
                'question_column_name': question_column_name,
                'context_column_name': context_column_name,
                'answer_column_name': answer_column_name
            },
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    metric = load_metric("squad")

    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments
        ) -> EvalPrediction:
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

        if training_args.do_predict:
            return formatted_predictions

        elif training_args.do_eval:
            references = [
                {"id": ex["id"], "answers": ex[answer_column_name]}
                for ex in datasets["validation"]
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )

    trainer = QATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=lambda x: metric.compute(predictions=x.predictions, references=x.label_ids),
    )

    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    logger.info("*** Evaluate ***")

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )
        logger.info("No metric can be presented because there is no correct answer given. Job done!")


if __name__ == "__main__":
    main()

