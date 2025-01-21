import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Literal, Optional, Tuple, Union

import torch
from data_processor import (
    llava_next_stage1_5_preprocessor,
    llava_next_stage2_preprocessor,
    llava_stage1_preprocessor,
    llava_stage2_preprocessor,
)
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from trainer import PackingImageCollator, PackingTrainer

from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    HfArgumentParser,
    PreTrainedTokenizer,
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
    data_preprocessor_type: Literal[
        "llava_stage-1.0",
        "llava_stage-2.0",
        "llava_next_stage-1.5",
        "llava_next_stage-2.0",
    ] = field(
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


@dataclass
class ImageTextToTextArguments(TrainingArguments, DataPipelineArguments, TrainPipelineArguments):
    output_dir: str = field(
        default=None,
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )

    def __post_init__(self):
        if self.output_dir is None:
            raise ValueError("output_dir은 무조건 설정되어 있어야 한다.")

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
    def processing_datasets(func: Callable) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        def process_dataset(
            dataset: Dataset,
            dataset_key: str,
            repo_name: str,
            truncate_map: dict,
            filter_cache_file_name: str,
        ) -> None:
            original_size = len(dataset)
            if dataset_key in truncate_map:
                truncate_size = truncate_map[dataset_key]
                dataset_size = len(dataset)
                dataset = dataset if dataset_size <= truncate_size else dataset.shuffle().select(range(truncate_size))
                if dataset_size <= truncate_size and train_args.is_world_process_zero:
                    logger.info(
                        f"{repo_name}의 {dataset_key}크기는 {dataset_size}이지만 truncate_size는 {truncate_size} 크기를 조절하셈."
                    )

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

        def concat(datasets_ls: List[Dataset], dataset_type: str) -> Optional[Dataset]:
            if datasets_ls:
                dataset = concatenate_datasets(datasets_ls)
                dataset.set_format("pt")
                if train_args.is_world_process_zero:
                    logger.info(f"{dataset_type}_dataset:\n{dataset}")
                return dataset
            return None

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
                process_dataset(datasets[dataset_key], dataset_key, repo_name, truncate_map, filter_cache_file_name)

        train_dataset = concat(train_dataset_ls, "train")
        valid_dataset = concat(valid_dataset_ls, "valid")
        test_dataset = concat(test_dataset_ls, "test")

        sample_dataset = train_dataset or valid_dataset or test_dataset
        if sample_dataset and train_args.is_world_process_zero:
            formated_instruct = processor.decode(sample_dataset[0]["input_ids"], skip_special_tokens=False)
            logger.info(f"formated_instruct: {formated_instruct}")

            if train_args.response_template is not None:
                response_template = processor.decode(train_args.response_template, skip_special_tokens=False)
                logger.info(f"response_template: {response_template}")
                if response_template not in formated_instruct:
                    raise ValueError("이거 response_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈")
            else:
                raise logger.error("response_template이 없음. 다시 설정하셈.")

            if train_args.instruction_template is not None:
                instruction_template = processor.decode(train_args.instruction_template, skip_special_tokens=False)
                logger.info(f"instruction_template: {instruction_template}")
                if instruction_template not in formated_instruct:
                    raise ValueError(
                        "이거 instruction_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈"
                    )
            else:
                logger.warning("instruction_template이 없음. 근데 애러는 발생하지 않고 그냥 패스함.")
        elif sample_dataset is None:
            logger.warning("train, valid, test데이터가 전혀 없는 상태인데 확인 한번 해봐.")

        if train_args.is_world_process_zero:
            logger.info(f"load_dataset_time: {time.time() - start_time:.2f}")

        return train_dataset, valid_dataset, test_dataset

    def check_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
        # copied from: transformers/models/llama/tokenization_llama.py:LlamaTokenizer:build_inputs_with_special_tokens()
        def build_inputs_with_special_tokens(tokenizer, token_ids_0, token_ids_1=None):
            bos_token_id = [tokenizer.bos_token_id] if tokenizer.add_bos_token else []
            eos_token_id = [tokenizer.eos_token_id] if tokenizer.add_eos_token else []

            output = bos_token_id + token_ids_0 + eos_token_id

            if token_ids_1 is not None:
                output = output + bos_token_id + token_ids_1 + eos_token_id

            return output

        input_ids = tokenizer("안녕하세요").input_ids
        bos_token_id, eos_token_id = tokenizer.bos_token_id, tokenizer.eos_token_id
        bos_token, eos_token = tokenizer.bos_token, tokenizer.eos_token
        is_add_bos, is_add_eos = input_ids[0] == bos_token_id, input_ids[-1] == eos_token_id

        user_chat = {"role": "user", "content": "Hello, how are you?"}
        assistant_chat = {"role": "assistant", "content": "I'm fine, thank you."}
        text = tokenizer.apply_chat_template([user_chat, assistant_chat], tokenize=False)

        msg = ""
        if not is_add_bos:
            msg += "tokenizer에 add_bos_token이 False로 되어 있음. 전처리 시, bos토큰이 삽입되지 않을 가능성이 있음.\n"
        if is_add_bos and bos_token in text:
            msg += "chat_template과 tokenizer에서 자동으로 bos추가해서 중복되어 들어갈 가능성이 있다.\n"
            is_add_bos = False

        if not is_add_eos:
            msg += "tokenizer에 add_eos_token이 False로 되어 있음. 전처리 시, eos토큰이 삽입되지 않을 가능성이 있음.\n"
        if is_add_eos and eos_token in text:
            msg += "chat_template과 tokenizer에서 자동으로 eos추가해서 중복되어 들어갈 가능성이 있다.\n"
            is_add_bos = False

        if train_args.is_world_process_zero:
            logger.warning(msg.strip())

        setattr(tokenizer, "add_bos_token", is_add_bos)
        setattr(tokenizer, "add_eos_token", is_add_eos)

        # TODO: 기존에 존재하던 build_inputs_with_special_tokens까지 덮어 씌어비리는 문제가 있다. 이거 나중에 채크해서 수정해야 할 듯.
        setattr(tokenizer, "build_inputs_with_special_tokens", build_inputs_with_special_tokens)

        return tokenizer

    # load model
    processor = AutoProcessor.from_pretrained(train_args.model_name_or_path, **train_args.processor_kwargs)
    config = AutoConfig.from_pretrained(train_args.model_name_or_path, **train_args.config_kwargs)

    model_kwargs = {"config": config, **train_args.model_kwargs}
    model = AutoModelForImageTextToText.from_pretrained(train_args.model_name_or_path, **model_kwargs)

    processor.tokenizer = check_tokenizer(processor.tokenizer)

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

    match train_args.data_preprocessor_type:
        case "llava_stage-1.0":
            preprocessor_func = llava_stage1_preprocessor
        case "llava_stage-2.0":
            preprocessor_func = llava_stage2_preprocessor
        case "llava_next_stage-1.5":
            preprocessor_func = llava_next_stage1_5_preprocessor
        case "llava_next_stage-2.0":
            preprocessor_func = llava_next_stage2_preprocessor

    context = (
        train_args.main_process_first(desc="main_process_first")
        if train_args.do_data_main_process_first
        else nullcontext()
    )

    with context:
        # load datasets
        train_dataset, valid_dataset, test_dataset = processing_datasets(preprocessor_func)

    # load collator
    collator = PackingImageCollator(
        tokenizer=processor.tokenizer,
        response_template=train_args.response_template,
        instruction_template=train_args.instruction_template,
        dtype=model.dtype,
    )

    # collator output check
    sample_check = collator.torch_call([[train_dataset[0]]] if train_args.do_packing else [train_dataset[0]])
    if train_args.is_world_process_zero:
        sample_check["labels"] = sample_check["labels"][sample_check["labels"] != -100].tolist()
        check_labels = [processor.tokenizer.convert_ids_to_tokens(token) for token in sample_check["labels"]]
        check_labels = ", ".join(check_labels)
        logger.info(f"collator_label: [-100,  ..., -100, {check_labels}]")

    if processor.tokenizer.bos_token_id not in sample_check.input_ids[0].tolist():
        raise ValueError("BOS token이 없다. 이거 다시 전처리 해라.")

    if processor.tokenizer.eos_token_id not in sample_check.input_ids[0].tolist():
        raise ValueError("EOS token이 없다. 이거 다시 전처리 해라.")

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
        setproctitle(train_args.run_name)

    main(train_args)
