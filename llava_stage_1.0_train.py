import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from transformers import (
    HfArgumentParser,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import is_main_process
from transformers.utils import is_liger_kernel_available


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class LlavaPretrainingArguments(TrainingArguments):
    # data
    dataset_repo_ls: List[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
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
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    valid_dataset_prefix: List[str] = field(
        default="validation",
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    test_dataset_prefix: List[str] = field(
        default="eval_other",
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    data_truncate_map: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "A map to truncate part of the data. {'repo_name': {'train': 3000, 'validation': 1500}}."},
    )
    data_name_map: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "A map to config_name of the data. {'repo_name': 'data_config_name'"},
    )

    cache_file_name: Optional[str] = field(
        default="preprocessor.arrow",
        metadata={"help": "Path to cached file name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    data_max_length: int = field(
        default=400,
        metadata={"help": "filtering max length dataset"},
    )

    sot_token: Optional[str] = field(
        default=None,
        metadata={"help": ""},
    )
    eot_token: Optional[str] = field(
        default=None,
        metadata={"help": ""},
    )
    do_data_main_process_first: bool = field(
        default=False,
        metadata={"help": "main process first"},
    )

    # model
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )

    response_template: str = field(
        default=None,
        metadata={"help": ""},
    )
    instruction_template: str = field(
        default=None,
        metadata={"help": "trl collator에서 사용되는 template 값."},
    )
    vision_feature_select_strategy: str = field(
        default="defualt",
        metadata={"help": "vision_feature_select_strategy에 사용되는 값, default, full 둘중에 하나만 고르셈."},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "어떤 attention 연산 방식을 사용할지 결정하는 값, default가 eager임, eager, flash_attention_2, sdpa중 하나 고르셈."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        self.data_truncate_map = json.loads(self.data_truncate_map) if self.data_truncate_map else {}
        self.data_name_map = json.loads(self.data_name_map) if self.data_name_map else {}
        self.response_template = json.loads(self.response_template) if self.response_template else None
        self.instruction_template = json.loads(self.instruction_template) if self.instruction_template else None

        self.train_dataset_prefix = self.train_dataset_prefix if self.train_dataset_prefix else []
        self.valid_dataset_prefix = self.valid_dataset_prefix if self.valid_dataset_prefix else []
        self.test_dataset_prefix = self.test_dataset_prefix if self.test_dataset_prefix else []

        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None


def main(train_args: LlavaPretrainingArguments) -> None:
    def preprocessor(example):
        if "caption" in example:
            conversation_ls = list()
            for caption in example["caption"]:
                conversation = [
                    {"role": "user", "content": json.dumps([{"type": "image"}], ensure_ascii=False)},
                    {
                        "role": "assistant",
                        "content": json.dumps([{"type": "text", "text": caption}], ensure_ascii=False),
                    },
                ]
                conversation_ls.append(conversation)

            example["conversations"] = conversation_ls

        finish_pixel_value_ls, finish_input_id_ls, finish_length_ls = list(), list(), list()
        for image, conversations in zip(example["image"], example["conversations"]):
            for chat in conversations:
                content = json.loads(chat["content"])
                chat["content"] = content

            image = image.convert("RGB")
            text = processor.apply_chat_template(
                conversations,
                tokenize=False,
                img_token=processor.image_token,
                sot_token=train_args.sot_token,
                eot_token=train_args.eot_token,
            )
            outputs = processor(text=text, images=image, return_tensors="np")

            pixel_values, input_ids, length = (
                outputs["pixel_values"][0],
                outputs["input_ids"][0],
                outputs["input_ids"][0].shape[0],
            )

            if image and (config.image_token_index not in input_ids):
                logger.info(f"text: {text}")
                logger.info(f"image: {image}")
                logger.info(f"input_ids: {input_ids}")
                logger.info(f"length: {length}")
                logger.info(f"image and (config.image_token_index not in input_ids) 필터링 됨.")
                break

            finish_pixel_value_ls.append(pixel_values)
            finish_input_id_ls.append(input_ids)
            finish_length_ls.append(length)

        return {
            "pixel_values": finish_pixel_value_ls,
            "input_ids": finish_input_id_ls,
            train_args.length_column_name: finish_length_ls,
        }

    def length_filter(length_ls):
        return [length <= train_args.data_max_length for length in length_ls]

    def prepare_datasets() -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        train_dataset_ls, valid_dataset_ls, test_dataset_ls = list(), list(), list()
        for repo_name in train_args.dataset_repo_ls:
            start_time = time.time()

            if is_main_process(train_args.local_rank):
                logger.info(f"load-{repo_name}")

            data_name = train_args.data_name_map.get(repo_name, None)
            truncate_map = train_args.data_truncate_map.get(repo_name, {})

            datasets = load_dataset(repo_name, data_name)

            map_cache_file_name = None
            filter_cache_file_name = None
            if train_args.cache_file_name:
                name = repo_name.split("/")[-1]
                name = f"{name}-{data_name}" if data_name else name

                map_cache_file_name = {
                    x: train_args.cache_dir.joinpath(f"map_{name}-{x}_{train_args.cache_file_name}").as_posix()
                    for x in datasets
                }
                filter_cache_file_name = {
                    x: train_args.cache_dir.joinpath(
                        f"filter_{train_args.data_max_length}_{name}-{x}_{train_args.cache_file_name}"
                    ).as_posix()
                    for x in datasets
                }

            # DatasetsDict이라서 이런식으로 해줘야 함.
            datasets = datasets.map(
                preprocessor,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                cache_file_names=map_cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=set(sum(datasets.column_names.values(), [])),
                desc=f"preprocess-{repo_name}",
            )

            if is_main_process(train_args.local_rank):
                logger.info(f"{repo_name}-before_filtering: {datasets}")

            datasets = datasets.filter(
                length_filter,
                num_proc=train_args.preprocessing_num_workers,
                input_columns=[train_args.length_column_name],
                cache_file_names=filter_cache_file_name,
                batched=train_args.preprocessing_batched,
                batch_size=train_args.preprocessing_batch_size,
                desc=f"length-filtering-{repo_name}",
            )

            for data_type in truncate_map:
                truncate_size = truncate_map[data_type]
                data = datasets[data_type].shuffle()
                if len(data) <= truncate_size:
                    if is_main_process(train_args.local_rank):
                        logger.info(
                            f"{repo_name}의 {data_type}크기는 {len(data)}이지만"
                            f"truncate_size는 {truncate_size} 크기를 조절하셈."
                        )
                    continue

                datasets[data_type] = data.select(range(truncate_size))

            if is_main_process(train_args.local_rank):
                logger.info(f"{repo_name}-after_filtering: {datasets}")
                logger.info(f"{repo_name}-load time: {time.time() - start_time}")

            for dataset_key in datasets:
                dataset = None
                if dataset_key in train_args.train_dataset_prefix and train_args.do_train:
                    dataset = datasets[dataset_key]
                    train_dataset_ls.append(dataset)

                if dataset_key in train_args.valid_dataset_prefix and train_args.do_eval:
                    dataset = datasets[dataset_key]
                    valid_dataset_ls.append(dataset)

                if dataset_key in train_args.test_dataset_prefix and train_args.do_predict:
                    dataset = datasets[dataset_key]
                    test_dataset_ls.append(dataset)

                if dataset and is_main_process(train_args.local_rank):
                    length_ls = sorted(dataset[train_args.length_column_name], reverse=True)[:100]
                    logger.info(f"{repo_name}/{dataset_key}-length: {length_ls}")

        train_dataset = None
        if train_dataset_ls:
            train_dataset = concatenate_datasets(train_dataset_ls)
            train_dataset.set_format("pt")
            if is_main_process(train_args.local_rank):
                logger.info(f"train_dataset:\n{train_dataset}")

        valid_dataset = None
        if valid_dataset_ls:
            valid_dataset = concatenate_datasets(valid_dataset_ls)
            valid_dataset.set_format("pt")
            if is_main_process(train_args.local_rank):
                logger.info(f"valid_dataset:\n{valid_dataset}")

        test_dataset = None
        if test_dataset_ls:
            test_dataset = concatenate_datasets(test_dataset_ls)
            test_dataset.set_format("pt")
            if is_main_process(train_args.local_rank):
                logger.info(f"test_dataset:\n{test_dataset}")

        sample_dataset = train_dataset or valid_dataset or test_dataset
        if sample_dataset and is_main_process(train_args.local_rank):
            formated_instruct = processor.decode(sample_dataset[0]["input_ids"], skip_special_tokens=False)
            response_template = processor.decode(train_args.response_template or [], skip_special_tokens=False)
            instruction_template = processor.decode(train_args.instruction_template or [], skip_special_tokens=False)

            if is_main_process(train_args.local_rank):
                logger.info(f"formated_instruct: {formated_instruct}")
                logger.info(f"response_template: {response_template}")
                logger.info(f"instruction_template: {instruction_template}")

            if train_args.do_train and train_args.response_template and response_template not in formated_instruct:
                raise ValueError("이거 response_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈")
            elif (
                train_args.do_train
                and train_args.instruction_template
                and instruction_template not in formated_instruct
            ):
                raise ValueError("이거 instruction_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈")
        elif sample_dataset is None:
            logger.warning("train, valid, test데이터가 전혀 없는 상태인데 확인 한번 해봐.")

        return (train_dataset, valid_dataset, test_dataset)

    # load model
    model_name_or_path = train_args.resume_from_checkpoint or train_args.model_name_or_path
    config = LlavaConfig.from_pretrained(
        model_name_or_path,
        attn_implementation=train_args.attn_implementation,
        vision_feature_select_strategy=train_args.vision_feature_select_strategy,
    )
    config.text_config.use_cache = False
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = LlavaProcessor.from_pretrained(model_name_or_path)

    if hasattr(processor, "vision_feature_use_cls") and "siglip" in config.vision_config.model_type:
        logger.info("이거 애러 방지하기 위한 임시 brench 사용하고 있음!!!!!!!!!!!!! 나중에 무조건 제거해!!!\n" * 10)
        tmp_cache_dir = train_args.cache_dir.joinpath("temp_fix")
        tmp_cache_dir.mkdir(exist_ok=True)
        train_args.cache_dir = tmp_cache_dir
        processor.vision_feature_use_cls = False

    logger.info(f"before_alive_param: {get_model_param_count(model, trainable_only=True)}")

    for name, parameter in model.named_parameters():
        name = name.split(".")[0]
        if name in ["language_model", "vision_tower"]:
            parameter.requires_grad = False

    logger.info(f"after_alive_param: {get_model_param_count(model, trainable_only=True)}")

    if is_liger_kernel_available() and train_args.use_liger_kernel:
        logger.info("now you use liger kernel!")
        from liger_kernel.transformers.trainer_integration import _apply_liger_kernel

        _apply_liger_kernel(model.language_model.config.model_type)

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    if train_args.do_data_main_process_first:
        with train_args.main_process_first(desc="main_process_first"):
            # load datasets
            train_dataset, valid_dataset, test_dataset = prepare_datasets()
    else:
        train_dataset, valid_dataset, test_dataset = prepare_datasets()

    # load collator
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=processor.tokenizer,
        response_template=train_args.response_template,
    )

    # load trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        processing_class=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    if train_args.do_train and train_dataset:
        train(trainer)

    if train_args.do_eval and valid_dataset:
        valid(trainer)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: Trainer) -> None:
    train_args: LlavaPretrainingArguments = trainer.args
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)


@torch.no_grad()
def valid(trainer: Trainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


if "__main__" in __name__:
    parser = HfArgumentParser([LlavaPretrainingArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if remain_args and is_main_process(train_args.local_rank):
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    main(train_args)
