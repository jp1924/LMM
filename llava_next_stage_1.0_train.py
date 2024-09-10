import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from transformers import (
    HfArgumentParser,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import logging as hf_logging


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class LlavaPretrainingArguments(TrainingArguments):
    # data
    dataset_repo_ls: List[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )

    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batched: bool = field(
        default=True,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    train_dataset_prefix: List[str] = field(
        default="train",
        metadata={"help": ""},
    )
    valid_dataset_prefix: List[str] = field(
        default="validation",
        metadata={"help": ""},
    )
    test_dataset_prefix: List[str] = field(
        default="eval_other",
        metadata={"help": ""},
    )

    data_truncate_map: Optional[Union[dict, str]] = field(default=None)

    cache_file_name: str = field(
        default=None,
        metadata={"help": "Path to cached file name"},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    # model
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )

    def __post_init__(self):
        super().__post_init__()
        self.data_truncate_map = json.loads(self.data_truncate_map) if self.data_truncate_map else None


def main(train_args: LlavaPretrainingArguments) -> None:
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        try:
            image_ls = example["image"]
            image_ls = image_ls if isinstance(image_ls, list) else [image_ls]
        except BaseException as e:
            logger.info(f"image load시 애러 발생: {e}")
            return {
                "pixel_values": [],
                "input_ids": [],
                train_args.length_column_name: [],
            }
        final_conver_ls = list()
        if "caption" in example:
            caption_ls = example["caption"]
            caption_ls = caption_ls if isinstance(caption_ls, list) else [caption_ls]
            for caption in caption_ls:
                conversation = [
                    {"role": "user", "content": [{"type": "image"}]},
                    {"role": "assistant", "content": [{"type": "text", "text": caption}]},
                ]
                final_conver_ls.append(conversation)
        elif "conversations" in example:
            conversations_ls = example["conversations"]
            conversations_ls = conversations_ls if isinstance(conversations_ls, list) else [conversations_ls]
            for idx, conversations in enumerate(conversations_ls):
                new_conversations = list()
                for chat in conversations:
                    chat["content"] = json.loads(chat["content"])
                    new_conversations.append(chat)
                conversations_ls[idx] = new_conversations
            final_conver_ls.extend(conversations_ls)
        elif "question_answer" in example:
            question_answer_ls = example["question_answer"]
            question_answer_ls = question_answer_ls if isinstance(question_answer_ls, list) else [question_answer_ls]
            for question_answers in question_answer_ls:
                question_answer = random.choice(question_answers)
                conversation = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": question_answer["question"]}, {"type": "image"}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": question_answer["answer"]}]},
                ]
                final_conver_ls.append(conversation)
        else:
            exit("지원하는 않는 데이터 타입, 종료함.")

        pixel_value_ls = list()
        input_id_ls = list()
        length_ls = list()
        for image, conversation in zip(image_ls, final_conver_ls):
            outputs = processor(
                images=image,
                text=processor.apply_chat_template(conversation, img_token=img_token),
                return_tensors="np",
            )
            pixel_value_ls.append(outputs["pixel_values"][0])
            input_id_ls.append(outputs["input_ids"][0])
            length_ls.append(outputs["input_ids"][0].shape[0])

        return {
            "pixel_values": pixel_value_ls,
            "input_ids": input_id_ls,
            train_args.length_column_name: length_ls,
        }

    def prepare_datasets() -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        train_dataset_ls = valid_dataset_ls = test_dataset_ls = list()
        for repo_name in train_args.dataset_repo_ls:
            logger.info(f"load-{repo_name}")
            datasets = load_dataset(repo_name)

            if repo_name in train_args.data_truncate_map:
                for data_type in train_args.data_truncate_map[repo_name]:
                    truncate_size = train_args.data_truncate_map[repo_name][data_type]
                    data = datasets[data_type].shuffle()
                    if len(data) <= truncate_size:
                        continue

                    datasets[data_type] = data.select(range(truncate_size))

            if train_args.cache_file_name:
                get_cache_path: str = lambda x: os.path.join(  # noqa: E731
                    train_args.cache_dir,
                    f"{name}-{x}_{train_args.cache_file_name}",
                )
                name = repo_name.split("/")[-1]
                train_args.cache_file_name = {x: get_cache_path(x) for x in datasets}

            # DatasetsDict이라서 이런식으로 해줘야 함.
            with train_args.main_process_first(desc="data preprocess"):
                datasets = datasets.map(
                    preprocessor,
                    num_proc=train_args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    batched=train_args.preprocessing_batched,
                    cache_file_names=train_args.cache_file_name,
                    batch_size=train_args.preprocessing_batch_size,
                    remove_columns=set(sum(datasets.column_names.values(), [])),
                    desc=f"preprocess-{repo_name}",
                )
                datasets.set_format("pt")
            for dataset_key in datasets:
                if dataset_key in train_args.train_dataset_prefix and train_args.do_train:
                    train_dataset_ls.append(datasets[dataset_key])
                if dataset_key in train_args.valid_dataset_prefix and train_args.do_eval:
                    valid_dataset_ls.append(datasets[dataset_key])
                if dataset_key in train_args.test_dataset_prefix and train_args.do_predict:
                    test_dataset_ls.append(datasets[dataset_key])

        train_dataset = None
        if train_dataset_ls:
            train_dataset = concatenate_datasets(train_dataset_ls)
            if train_args.local_rank <= 0:
                logger.info(f"train_dataset:\n{train_dataset}")

        valid_dataset = None
        if valid_dataset_ls:
            valid_dataset = concatenate_datasets(valid_dataset_ls)
            if train_args.local_rank <= 0:
                logger.info(f"valid_dataset:\n{valid_dataset}")

        test_dataset = None
        if test_dataset_ls:
            test_dataset = concatenate_datasets(test_dataset_ls)
            if train_args.local_rank <= 0:
                logger.info(f"test_dataset:\n{test_dataset}")

        return (train_dataset, valid_dataset, test_dataset)

    # load model
    model_name_or_path = train_args.resume_from_checkpoint or train_args.model_name_or_path
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = LlavaProcessor.from_pretrained(model_name_or_path)

    img_token = processor.tokenizer.convert_ids_to_tokens(model.config.image_token_index)

    for name, parameter in model.named_parameters():
        name = name.split(".")[0]
        if name not in ["language_model"]:
            continue
        parameter.requires_grad = False

    # load dataset & preprocess
    train_dataset, valid_dataset, test_dataset = prepare_datasets()

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    response_template = processor.tokenizer.encode("\n\n### Assistant:\n", add_special_tokens=False)[3:]
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=processor.tokenizer,
        response_template=response_template,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )
    if train_args.do_train and train_dataset:
        train(trainer)

    if train_args.do_eval and valid_dataset:
        valid(trainer)

    if train_args.do_predict and test_dataset:
        predict(trainer, test_dataset)


def train(trainer: Trainer) -> None:
    train_args: LlavaPretrainingArguments = trainer.args
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)


@torch.no_grad()
def valid(trainer: Trainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


@torch.no_grad()
def predict(trainer: Trainer, test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    test_dataset_dict = dict()
    test_name_ls = test_dataset["dataset_name"]
    for dataset_name in set(test_name_ls):
        part_idx = [idx for idx, x in enumerate(test_name_ls) if x == dataset_name]
        part_dataset = test_dataset.select(part_idx, keep_in_memory=False)

        # 'jp1924/KconfSpeech-validation'
        start = dataset_name.rindex("/") + 1
        end = dataset_name.rindex("-")

        outputs = trainer.predict(part_dataset, metric_key_prefix=f"test/{dataset_name[start:]}")
        # NOTE: trainer.log를 사용하면 train/test 처럼 찍혀서 나와서 wandb로 직접 찍음
        if GLOBAL_LOGGER:
            GLOBAL_LOGGER.log(outputs.metrics)
        test_dataset_dict[dataset_name[start:end]] = part_dataset


if "__main__" in __name__:
    parser = HfArgumentParser([LlavaPretrainingArguments])
    train_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    main(train_args)
