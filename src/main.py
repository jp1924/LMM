import json
import logging
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import optimization
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from datasets import logging as ds_logging
from preprocessor import PROCESSOR_REGISTRY
from setproctitle import setproctitle
from trainer import PackingImageCollatorForCompletionOnlyLM, PackingTrainer

from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
    TrainingArguments,
)
from transformers import logging as hf_logging
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import set_seed
from transformers.utils import is_sagemaker_mp_enabled


@dataclass
class DataPipelineArguments:
    # data
    dataset_repo_ls: List[str] = field(
        default_factory=list,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    data_preprocessor_type: str = field(
        default_factory=str,
        metadata={
            "help": "preprocessor type, [llava_stage-1.0, llava_stage-2.0, llava_next_stage-1.5, llava_next_stage-2.0]"
        },
    )
    do_data_main_process_first: bool = field(
        default=False,
        metadata={"help": "main process first"},
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
        default_factory=list,
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    valid_dataset_prefix: List[str] = field(
        default_factory=list,
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    test_dataset_prefix: List[str] = field(
        default_factory=list,
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )

    data_truncate_map: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": "A map to truncate part of the data. {'repo_name': {'train': 3000, 'validation': 1500}}."},
    )
    data_name_map: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": "A map to config_name of the data. {'repo_name': 'data_config_name'"},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    data_max_length: int = field(
        default=2048,
        metadata={"help": "filtering max length dataset"},
    )


@dataclass
class TrainPipelineArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    response_template: str = field(
        default="[]",
        metadata={"help": "trl collator에서 사용되는 template 값."},
    )
    instruction_template: str = field(
        default="[]",
        metadata={"help": "trl collator에서 사용되는 template 값."},
    )
    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "어떤 attention 연산 방식을 사용할지 결정하는 값, default가 eager임, eager, flash_attention_2, sdpa중 하나 고르셈."
        },
    )
    vision_learning_rate: float = field(
        default=2e-6,
        metadata={"help": "오직 llava one vision 모델에서만 사용할 수 있다."},
    )
    packing_max_elem: int = field(
        default=10,
        metadata={"help": ""},
    )
    do_packing: bool = field(
        default=True,
        metadata={"help": ""},
    )
    packing_shuffle: bool = field(
        default=True,
        metadata={"help": "packing shuffle"},
    )
    freeze_named_param: List[str] = field(
        default=None,
        metadata={"help": "freeze_named_param"},
    )
    profiling: bool = field(
        default=False,
        metadata={"help": "profiling"},
    )
    profiling_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": "profiling_kwargs"},
    )
    config_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    model_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    processor_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": ""},
    )
    padding_side: str = field(
        default="left",
        metadata={"help": ""},
    )


@dataclass
class ImageTextToTextArguments(TrainingArguments, DataPipelineArguments, TrainPipelineArguments):
    def __post_init__(self):
        super().__post_init__()

        def _convert_str_dict(passed_value: dict):
            "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
            for key, value in passed_value.items():
                if isinstance(value, dict):
                    passed_value[key] = _convert_str_dict(value)
                elif isinstance(value, str):
                    # First check for bool and convert
                    if value.lower() in ("true", "false"):
                        passed_value[key] = value.lower() == "true"
                    # Check for digit
                    elif value.isdigit():
                        passed_value[key] = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        passed_value[key] = float(value)

            return passed_value

        _ADDITIONAL_VALID_DICT_FILEDS = [
            "data_truncate_map",
            "data_name_map",
            "config_kwargs",
            "model_kwargs",
            "processor_kwargs",
            "profiling_kwargs",
        ]
        _VALID_LIST_FIELDS = [
            "instruction_template",
            "response_template",
            "train_dataset_prefix",
            "valid_dataset_prefix",
            "test_dataset_prefix",
            "freeze_named_param",
        ]

        # copied from: transformers/training_args.py/__post_init__()
        for field in _ADDITIONAL_VALID_DICT_FILEDS:
            passed_value = getattr(self, field)
            # We only want to do this if the str starts with a bracket to indiciate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, field, loaded_dict)
            elif isinstance(passed_value, dict) or passed_value is None:
                pass
            else:
                raise ValueError(f"{field}은 dict로 설정해야 함.")

        for field in _VALID_LIST_FIELDS:
            passed_value = getattr(self, field)
            if isinstance(passed_value, str) and passed_value.startswith("["):
                loaded_list = json.loads(passed_value)
                setattr(self, field, loaded_list)
            elif isinstance(passed_value, list) or passed_value is None:
                pass
            else:
                raise ValueError(f"{field}은 list로 설정해야 함.")

        self.config_kwargs = {
            **self.config_kwargs,
            "attn_implementation": self.attn_implementation,
        }

        self.processor_kwargs = {
            **self.processor_kwargs,
            "padding_side": self.padding_side,
        }

        if self.chat_template:
            self.tokenizer_kwargs["chat_template"] = self.chat_template

        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None
        self.model_name_or_path = self.resume_from_checkpoint or self.model_name_or_path

        if self.group_by_length:
            logger.warning("group_by_length이 True임! loss계산에 영향을 끼칠 수 있으니 확인해.")

    @property
    def is_local_process_zero(self) -> bool:
        return self.local_process_index == 0

    @property
    def is_world_process_zero(self) -> bool:
        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp  # type: ignore

            return smp.rank() == 0
        else:
            return self.process_index == 0


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def main(train_args: ImageTextToTextArguments) -> None:
    def processing_datasets(
        func: Callable,
    ) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        def process_dataset(dataset, dataset_key, repo_name, truncate_map, filter_cache_file_name):
            original_size = len(dataset)

            if dataset_key in truncate_map:
                truncate_size = truncate_map[dataset_key]
                dataset_size = len(dataset)
                dataset = dataset if dataset_size <= truncate_size else dataset.shuffle().select(range(truncate_size))
                if dataset_size <= truncate_size and train_args.is_world_process_zero:
                    logger.info(
                        f"{repo_name}의 {dataset_key}크기는 {dataset_size}이지만 truncate_size는 {truncate_size} 크기를 조절하셈."
                    )

            if train_args.is_world_process_zero:
                range_histogram(dataset["length"], 100, 50)

            if dataset_key in train_args.train_dataset_prefix and train_args.do_train:
                dataset = dataset.filter(
                    lambda length_ls: [length <= train_args.data_max_length for length in length_ls],  # type: ignore
                    num_proc=train_args.preprocessing_num_workers,
                    input_columns=[train_args.length_column_name],
                    cache_file_name=filter_cache_file_name[dataset_key],
                    batched=train_args.preprocessing_batched,
                    batch_size=train_args.preprocessing_batch_size,
                    desc=f"length-filtering-{repo_name}/{dataset_key}",
                )
                train_dataset_ls.append(dataset)

            if dataset_key in train_args.valid_dataset_prefix and train_args.do_eval:
                valid_dataset_ls.append(dataset)

            if dataset_key in train_args.test_dataset_prefix and train_args.do_predict:
                test_dataset_ls.append(dataset)

            if train_args.is_world_process_zero:
                length_ls = sorted(dataset[train_args.length_column_name], reverse=True)[:100]
                length_ls = [int(length) for length in length_ls]
                logger.info(f"{repo_name}/{dataset_key}-length: {length_ls}")
                logger.info(f"{repo_name}/{dataset_key}-size: {original_size} -> {len(dataset)}")

        def concat(datasets_ls, dataset_type):
            if datasets_ls:
                dataset = concatenate_datasets(datasets_ls)
                dataset.set_format("pt")
                if train_args.is_world_process_zero:
                    logger.info(f"{dataset_type}_dataset:\n{dataset}")
                return dataset
            return None

        def range_histogram(data, num_bins=50, width=50):
            # 데이터의 최대값과 최소값 찾기
            min_val = min(data)
            max_val = max(data)

            # 구간 크기 계산
            bin_size = (max_val - min_val) / num_bins

            # 각 구간별 빈도수 계산
            bins = [0] * num_bins
            for value in data:
                bin_index = min(int((value - min_val) / bin_size), num_bins - 1)
                bins[bin_index] += 1

            # 최대 빈도수 찾기
            max_freq = max(bins)

            # 히스토그램 출력
            logger.info(f"\nHistogram (total {len(data)} items, {num_bins} bins)")
            logger.info("-" * 80)
            logger.info(f"Range{' ' * 18}Count  Distribution")
            logger.info("-" * 80)

            for i in range(num_bins):
                start = min_val + (i * bin_size)
                end = min_val + ((i + 1) * bin_size)
                bar_length = int((bins[i] / max_freq) * width)
                bar = "█" * bar_length

                # 구간과 빈도수, 막대 출력
                logger.info(f"{start:8.0f}-{end:8.0f}: {bins[i]:6d} |{bar}")

            logger.info("-" * 80)
            logger.info("\nStatistics:")
            logger.info(f"데이터 개수: {len(data)}")
            logger.info(f"최소값: {min_val:.0f}")
            logger.info(f"최대값: {max_val:.0f}")
            logger.info(f"평균값: {sum(data) / len(data):.2f}")
            logger.info(f"구간 크기: {bin_size:.2f}")

        start_time = time.time()
        train_dataset_ls, valid_dataset_ls, test_dataset_ls = [], [], []
        for repo_name in train_args.dataset_repo_ls:
            if train_args.is_world_process_zero:
                logger.info(f"load-{repo_name}")

            data_name = train_args.data_name_map.get(repo_name, None)
            truncate_map = train_args.data_truncate_map.get(repo_name, {})
            datasets = load_dataset(repo_name, data_name)

            map_cache_file_name, filter_cache_file_name = None, None
            if train_args.cache_dir is not None:
                name = repo_name.split("/")[-1]
                name = f"{name}-{data_name}" if data_name else name

                map_cache_file_name = {
                    x: train_args.cache_dir.joinpath(f"map_{name}-{x}_preprocessor.arrow").as_posix() for x in datasets
                }
                filter_cache_file_name = {
                    x: train_args.cache_dir.joinpath(
                        f"filter_{f'{truncate_map[x]}-' if x in truncate_map else ''}{train_args.data_max_length}_{name}-{x}_preprocessor.arrow"
                    ).as_posix()
                    for x in datasets
                }

            datasets = datasets.map(
                func,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                cache_file_names=map_cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=set(sum(datasets.column_names.values(), [])),
                desc=f"preprocess-{repo_name}",
                fn_kwargs={"processor": processor, "args": train_args},
            )

            for dataset_key in datasets:
                process_dataset(
                    datasets[dataset_key],
                    dataset_key,
                    repo_name,
                    truncate_map,
                    filter_cache_file_name,
                )

        train_dataset = concat(train_dataset_ls, "train")
        valid_dataset = concat(valid_dataset_ls, "valid")
        test_dataset = concat(test_dataset_ls, "test")

        if train_args.is_world_process_zero and train_dataset:
            logger.info("train-datasets")
            range_histogram(train_dataset["length"], 100, 50)
        if train_args.is_world_process_zero and valid_dataset:
            logger.info("valid-datasets")
            if isinstance(valid_dataset, dict):
                for key in valid_dataset:
                    range_histogram(valid_dataset[key]["length"], 100, 50)
            else:
                range_histogram(valid_dataset["length"], 100, 50)
        if train_args.is_world_process_zero and test_dataset:
            logger.info("test-datasets")
            if isinstance(test_dataset, dict):
                for key in test_dataset:
                    range_histogram(test_dataset[key]["length"], 100, 50)
            else:
                range_histogram(test_dataset["length"], 100, 50)

        if train_args.is_world_process_zero:
            logger.info(f"load_dataset_time: {time.time() - start_time:.2f}")

        return train_dataset, valid_dataset, test_dataset

    # load model
    processor = AutoProcessor.from_pretrained(train_args.model_name_or_path, **train_args.processor_kwargs)
    config = AutoConfig.from_pretrained(train_args.model_name_or_path, **train_args.config_kwargs)

    model_kwargs = {"config": config, **train_args.model_kwargs}

    with (
        train_args.main_process_first(desc="main_process_first")
        if train_args.do_data_main_process_first
        else nullcontext()
    ):
        # load datasets
        train_dataset, valid_dataset, test_dataset = processing_datasets(
            PROCESSOR_REGISTRY[train_args.data_preprocessor_type]
        )

        model = AutoModelForImageTextToText.from_pretrained(train_args.model_name_or_path, **model_kwargs)

    if train_args.freeze_named_param:
        freeze_param_ls = [param for name, param in model.named_parameters() if name in train_args.freeze_named_param]
        if not freeze_param_ls:
            raise ValueError("freeze_named_param에 해당하는 모듈이 없음.")

        for param in freeze_param_ls:
            param.requires_grad = False

        if train_args.is_world_process_zero:
            full_param_num = get_model_param_count(model, trainable_only=False)
            alive_param_num = get_model_param_count(model, trainable_only=True)
            dead_param_num = full_param_num - alive_param_num

            logger.info(
                f"얼린 파라미터 수: {dead_param_num}, 활성화된 파라미터 수: {alive_param_num}, 전체 파라미터 수: {full_param_num}"
            )

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    # load collator
    collator = PackingImageCollatorForCompletionOnlyLM(
        tokenizer=processor.tokenizer,
        args=train_args,
        sample_dataset=train_dataset or valid_dataset or test_dataset,
        dtype=model.dtype,
        response_template=train_args.response_template,
        instruction_template=train_args.instruction_template,
    )

    # load trainer
    trainer = PackingTrainer(
        model=model,
        args=train_args,
        processing_class=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    if train_args.do_train and train_dataset:
        train(trainer, train_args)

    if train_args.do_eval and valid_dataset:
        valid(trainer, valid_dataset)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: PackingTrainer, args: ImageTextToTextArguments) -> None:
    from accelerate import ProfileKwargs

    # profile_kwargs = ProfileKwargs(activities=["cpu", "cuda"], profile_memory=True, with_flops=True)
    context = trainer.accelerator.profile(ProfileKwargs(**args.profiling_kwargs)) if args.profiling else nullcontext()

    with context as prof:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    save_path = Path(args.output_dir)
    if prof:
        prof.export_memory_timeline(save_path.with_suffix(".memory_trace.json").as_posix())
        prof.export_chrome_trace(save_path.with_suffix(".chrome_trace.json").as_posix())
        print(prof.key_averages().table(sort_by="flops", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


@torch.no_grad()
def valid(trainer: PackingTrainer, valid_datasets: Dataset) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


if "__main__" in __name__:
    parser = HfArgumentParser([ImageTextToTextArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if remain_args and train_args.is_world_process_zero:
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(f"{train_args.run_name}-{train_args.local_process_index}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    ds_logging.set_verbosity(log_level)
    hf_logging.set_verbosity(log_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    main(train_args)
