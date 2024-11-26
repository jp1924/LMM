import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    LlavaNextConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.trainer_utils import is_main_process
from transformers.utils import is_liger_kernel_available


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class LlavaNextPretrainingArguments(TrainingArguments):
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
        metadata={"help": "A map to truncate part of the data. {‘repo_name’: {‘train’: 3000, ‘validation’: 1500}}."},
    )
    data_name_map: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "A map to config_name of the data. {'repo_name': 'data_config_name'"},
    )
    cache_file_name: Optional[str] = field(
        default=None,
        metadata={"help": "Path to cached file name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    data_max_length: int = field(
        default=3072,
        metadata={"help": "filtering max length dataset"},
    )

    sot_token: str = field(
        default=None,
        metadata={"help": "start of text token"},
    )
    eot_token: str = field(
        default=None,
        metadata={"help": "end of text token"},
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
    vision_learning_rate: float = field(
        default=2e-6,
        metadata={"help": "The initial learning rate for AdamW."},
    )
    response_template: str = field(
        default=None,
        metadata={"help": "trl collator에서 사용되는 template 값."},
    )
    instruction_template: str = field(
        default=None,
        metadata={"help": "trl collator에서 사용되는 template 값."},
    )
    image_grid_pinpoints: str = field(
        default=None,
        metadata={"help": "llava-next image processor에 사용되는 image_grid 사이즈"},
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
    profiling: bool = field(
        default=False,
        metadata={"help": "profiling"},
    )

    def __post_init__(self):
        super().__post_init__()
        self.data_truncate_map = json.loads(self.data_truncate_map) if self.data_truncate_map else {}
        self.data_name_map = json.loads(self.data_name_map) if self.data_name_map else {}
        self.response_template = json.loads(self.response_template) if self.response_template else None
        self.image_grid_pinpoints = json.loads(self.image_grid_pinpoints) if self.image_grid_pinpoints else None

        self.train_dataset_prefix = self.train_dataset_prefix if self.train_dataset_prefix else []
        self.valid_dataset_prefix = self.valid_dataset_prefix if self.valid_dataset_prefix else []
        self.test_dataset_prefix = self.test_dataset_prefix if self.test_dataset_prefix else []

        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None


class DataCollatorForImageCompletion(DataCollatorForCompletionOnlyLM):
    def __init__(self, image_processor, **kwargs):
        super().__init__(**kwargs)
        self.image_processor = image_processor

    def torch_call(self, examples):
        input_ids = [{"input_ids": example["input_ids"]} for example in examples]
        pixel_values = [example["pixel_values"] for example in examples if example["pixel_values"] is not None]
        image_sizes = [example["image_sizes"] for example in examples if example["image_sizes"] is not None]
        batch = super().torch_call(input_ids)

        if pixel_values:
            batch["pixel_values"] = torch.concat(pixel_values, dim=0)

        if image_sizes:
            batch["image_sizes"] = torch.stack(image_sizes, dim=0)

        return batch


class LLaVANextTrainer(Trainer):
    def create_optimizer(self):
        decay_parameters = self.get_decay_parameter_names(self.model)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.multi_modal_projector.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": train_args.weight_decay,
                "lr": train_args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.multi_modal_projector.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": train_args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.language_model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": train_args.weight_decay,
                "lr": train_args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.language_model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": train_args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.vision_tower.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": train_args.weight_decay,
                "lr": train_args.vision_learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.vision_tower.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": train_args.vision_learning_rate,
            },
        ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(train_args, self.model)

        if "params" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("params")

        # Overwrite `model` in case it's created by `get_optimizer_cls_and_kwargs`
        # e.g. for LOMO optimizer.
        if "model" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("model")

        # For layer-wise dummy optimizers we overwrite optimizer_grouped_parameters with `optimizer_dict`
        # to avoid arguments conflicts.
        if "optimizer_dict" in optimizer_kwargs:
            optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return optimizer


def main(train_args: LlavaNextPretrainingArguments) -> None:
    def preprocessor(example):
        finish_pixel_value_ls, finish_input_id_ls, finish_length_ls, finish_image_sizes_ls = (
            list(),
            list(),
            list(),
            list(),
        )
        for conversations, image in zip(example["conversations"], example["image"]):
            for chat in conversations:
                chat["content"] = json.loads(chat["content"])

            text = processor.apply_chat_template(
                conversations,
                tokenize=False,
                img_token=processor.image_token,
                sot_token=train_args.sot_token,
                eot_token=train_args.eot_token,
            )
            outputs = processor(text=text, images=image.convert("RGB"), return_tensors="np")

            pixel_values, input_ids, image_sizes, length = (
                outputs["pixel_values"][0],
                outputs["input_ids"][0],
                outputs["image_sizes"][0],
                outputs["input_ids"][0].shape[0],
            )

            if image and (config.image_token_index not in input_ids):
                logger.info(
                    f"text: {text}"
                    f"image: {image}"
                    f"input_ids: {input_ids}"
                    f"length: {length}"
                    "image and (config.image_token_index not in input_ids) 필터링 됨."
                )
                break
            elif (image is None) and (config.image_token_index in input_ids):
                logger.info(
                    f"text: {text}"
                    f"image: {image}"
                    f"input_ids: {input_ids}"
                    f"length: {length}"
                    "(image is None) and (config.image_token_index in input_ids) 필터링 됨."
                )
                break
            finish_pixel_value_ls.append(pixel_values)
            finish_image_sizes_ls.append(image_sizes)
            finish_input_id_ls.append(input_ids)
            finish_length_ls.append(length)

        return {
            "pixel_values": finish_pixel_value_ls,
            "image_sizes": finish_image_sizes_ls,
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

            # prprocessing datasets
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
                logger.info(datasets)
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

        # concatenate train, valid, test datasets and set format
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

        # check response_template, instruction_template
        sample_dataset = train_dataset or valid_dataset or test_dataset
        if sample_dataset and is_main_process(train_args.local_rank):
            response_template = getattr(train_args, "response_template", None)
            instruction_template = getattr(train_args, "instruction_template", None)
            formated_instruct = processor.decode(sample_dataset[0]["input_ids"], skip_special_tokens=False)
            logger.info(f"formated_instruct: {formated_instruct}")

            if response_template is not None:
                response_template = processor.decode(response_template, skip_special_tokens=False)
                logger.info(f"response_template: {response_template}")
                if response_template not in formated_instruct:
                    raise ValueError("이거 response_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈")
            elif not hasattr(train_args, "response_template"):
                logger.warning("train_args에 response_template이 없음. 근데 애러는 발생하지 않고 그냥 패스함.")
            else:
                raise logger.error("response_template이 없음. 다시 서정하셈.")

            if instruction_template is not None:
                instruction_template = processor.decode(instruction_template, skip_special_tokens=False)
                logger.info(f"instruction_template: {instruction_template}")
                if instruction_template not in formated_instruct:
                    raise ValueError(
                        "이거 instruction_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈"
                    )
            elif not hasattr(train_args, "instruction_template"):
                logger.warning("train_args에 response_template이 없음. 근데 애러는 발생하지 않고 그냥 패스함.")
            else:
                logger.warning("instruction_template이 없음. 근데 애러는 발생하지 않고 그냥 패스함.")
        elif sample_dataset is None:
            logger.warning("train, valid, test데이터가 전혀 없는 상태인데 확인 한번 해봐.")

        return (train_dataset, valid_dataset, test_dataset)

    # load model
    model_name_or_path = train_args.resume_from_checkpoint or train_args.model_name_or_path or ""
    config = LlavaNextConfig.from_pretrained(
        model_name_or_path,
        image_grid_pinpoints=train_args.image_grid_pinpoints,
        attn_implementation=train_args.attn_implementation,
        vision_feature_select_strategy=train_args.vision_feature_select_strategy,
    )
    config.text_config.use_cache = False
    model = LlavaNextForConditionalGeneration.from_pretrained(model_name_or_path, config=config)

    image_processor = LlavaNextImageProcessor.from_pretrained(
        model_name_or_path,
        image_grid_pinpoints=train_args.image_grid_pinpoints,
        crop_size={"height": config.vision_config.image_size, "width": config.vision_config.image_size},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    processor = LlavaNextProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        chat_template=tokenizer.chat_template,
        image_token=tokenizer.convert_ids_to_tokens(config.image_token_index),
        patch_size=config.vision_config.patch_size,
        vision_feature_select_strategy=train_args.vision_feature_select_strategy,
    )

    if is_liger_kernel_available() and train_args.use_liger_kernel:
        from liger_kernel.transformers.trainer_integration import _apply_liger_kernel

        text_model_type = model.language_model.config.model_type
        _apply_liger_kernel(text_model_type)

    if hasattr(processor, "vision_feature_use_cls") and "siglip" in config.vision_config.model_type:
        logger.info("이거 애러 방지하기 위한 임시 brench 사용하고 있음!!!!!!!!!!!!! 나중에 무조건 제거해!!!\n" * 10)
        tmp_cache_dir = train_args.cache_dir.joinpath("temp_fix")
        tmp_cache_dir.mkdir(exist_ok=True)
        train_args.cache_dir = tmp_cache_dir
        processor.vision_feature_use_cls = False

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    context = (
        train_args.main_process_first(desc="main_process_first")
        if train_args.do_data_main_process_first
        else nullcontext()
    )
    with context:
        # load datasets
        train_dataset, valid_dataset, test_dataset = prepare_datasets()

    collator = DataCollatorForImageCompletion(
        tokenizer=processor.tokenizer,
        image_processor=processor.image_processor,
        response_template=train_args.response_template,
        instruction_template=train_args.instruction_template,
    )

    # collator output check
    if is_main_process(train_args.local_rank):
        sample_check = collator.torch_call([train_dataset[0]])
        sample_check["labels"] = sample_check["labels"][sample_check["labels"] != -100].tolist()
        check_labels = [processor.tokenizer.convert_ids_to_tokens(token) for token in sample_check["labels"]]
        check_labels = ", ".join(check_labels)
        logger.info(rf"collator_label: [-100,  ..., -100, {check_labels}]")

    # load trainer
    trainer = LLaVANextTrainer(
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


def train(trainer: Trainer, args: LlavaNextPretrainingArguments) -> None:
    from accelerate import ProfileKwargs

    profile_kwargs = ProfileKwargs(activities=["cpu", "cuda"], profile_memory=True, with_flops=True)
    context = trainer.accelerator.profile(profile_kwargs) if args.profiling else nullcontext()

    with context as prof:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    save_path = Path(args.output_dir)
    if prof:
        prof.export_memory_timeline(save_path.with_suffix(".memory_trace.json").as_posix())
        prof.export_chrome_trace(save_path.with_suffix(".chrome_trace.json").as_posix())
        print(prof.key_averages().table(sort_by="flops", row_limit=10))
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))


@torch.no_grad()
def valid(trainer: Trainer, valid_datasets: Dataset) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


if "__main__" in __name__:
    parser = HfArgumentParser([LlavaNextPretrainingArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if train_args.response_template is None:
        exit("response_template가 무조건 설정되어 있어야 함!")

    if remain_args and is_main_process(train_args.local_rank):
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    main(train_args)
