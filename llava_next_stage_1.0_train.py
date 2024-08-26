import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

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
    is_torch_xla_available,
    is_wandb_available,
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
    valid_exclude_ls: List[str] = field(
        default="",
        metadata={"help": ""},
    )
    valid_truncate_num: int = field(
        default=3000,
        metadata={"help": ""},
    )
    split_valid: bool = field(
        default=False,
        metadata={"help": ""},
    )
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
    wandb_code_log_dir: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )

global GLOBAL_LOGGER
GLOBAL_LOGGER = None


def main(train_args: LlavaPretrainingArguments) -> None:
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        try:
            image_ls = example["image"]
            image_ls = image_ls if isinstance(image_ls, list) else [image_ls]
        except:
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
                question_answers = [x for x in question_answers if x["confidence"] == "true"]
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

    def collect_dataset(prefix_ls: List[str]) -> Optional[Dataset]:
        if not prefix_ls:
            return None

        data_ls = list()
        for prefix in prefix_ls:
            check_key: str = lambda key: (prefix in key)
            filter_data = [
                concatenate_datasets(data_dict.pop(key)) for key in list(data_dict.keys()) if check_key(key)
            ]
            data_ls.extend(filter_data)
        dataset = concatenate_datasets(data_ls)
        dataset.set_format("torch")

        return dataset

    def set_wandb() -> None:
        # TODO: 이건 나중에 args로 바꿀 것
        GLOBAL_LOGGER.run.log_code(
            train_args.wandb_code_log_dir,
            include_fn=lambda path: path.endswith(".py") or path.endswith(".json"),
        )
        # logging args
        combined_dict = {**train_args.to_dict()}
        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}

        GLOBAL_LOGGER.config.update(combined_dict, allow_val_change=True)

        # set default metrics
        if getattr(GLOBAL_LOGGER, "define_metric", None):
            GLOBAL_LOGGER.define_metric("train/global_step")
            GLOBAL_LOGGER.define_metric("*", step_metric="train/global_step", step_sync=True)

        # set model watch
        _watch_model = os.getenv("WANDB_WATCH", "false")
        if not is_torch_xla_available() and _watch_model in ("all", "parameters", "gradients"):
            GLOBAL_LOGGER.watch(model, log=_watch_model, log_freq=max(100, train_args.logging_steps))
        GLOBAL_LOGGER.run._label(code="transformers_trainer")

    # load model
    model_name_or_path = train_args.resume_from_checkpoint or train_args.model_name_or_path or ""
    model = LlavaForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = LlavaProcessor.from_pretrained(model_name_or_path)

    img_token = processor.tokenizer.convert_ids_to_tokens(model.config.image_token_index)

    for name, parameter in model.named_parameters():
        name = name.split(".")[0]
        if name not in ["language_model"]:
            continue
        parameter.requires_grad = False

    # set logger
    if GLOBAL_LOGGER and (train_args.local_rank == 0):
        set_wandb()

    # load dataset & preprocess
    data_dict = dict()
    for dataset_name in train_args.dataset_repo_ls:
        logger.info(f"load-{dataset_name}")
        dataset = load_dataset(dataset_name)

        # DatasetDict이라서 이런식으로 해줘야 함.
        column_names = set(sum(dataset.column_names.values(), []))
        with train_args.main_process_first(desc="data preprocess"):
            cache_file_name = None
            if train_args.cache_file_name:
                get_cache_path: str = lambda x: os.path.join(
                    train_args.cache_dir,
                    f"{name}-{x}_{train_args.cache_file_name}",
                )
                name = dataset_name.split("/")[-1]
                cache_file_name = {x: get_cache_path(x) for x in dataset}

            dataset = dataset.map(
                preprocessor,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                cache_file_names=cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=column_names,
                desc=f"preprocess-{dataset_name}",
            )

        for data_key in dataset:
            if data_key not in data_dict:
                data_dict[data_key] = []

            specific_dataset = dataset[data_key]

            added_data = [f"{dataset_name}-{data_key}"] * len(specific_dataset)
            specific_dataset = specific_dataset.add_column("dataset_name", added_data)

            data_dict[data_key].append(specific_dataset)

    train_dataset = None
    if train_args.do_train:
        train_dataset = collect_dataset(train_args.train_dataset_prefix)
        if (train_args.local_rank == 0) and train_dataset:
            logger.info("train_dataset")
            logger.info(train_dataset)

    valid_dataset = None
    if train_args.do_eval:
        valid_dataset = collect_dataset(train_args.valid_dataset_prefix)

        valid_dataset_dict = dict()
        valid_repo_ls = valid_dataset["dataset_name"]
        valid_exclude_ls = train_args.valid_exclude_ls or []
        if train_args.split_valid:
            for dataset_name in set(valid_repo_ls):
                part_idx = [idx for idx, x in enumerate(valid_repo_ls) if x == dataset_name]
                part_dataset = valid_dataset.select(part_idx, keep_in_memory=False)

                # 'jp1924/KconfSpeech-validation'
                start = dataset_name.rindex("/") + 1
                end = dataset_name.rindex("-")

                if dataset_name[start:end] in valid_exclude_ls:
                    continue

                if len(part_dataset) > train_args.valid_truncate_num:
                    part_dataset = part_dataset.shuffle(train_args.seed)
                    part_dataset = part_dataset.select(range(train_args.valid_truncate_num))

                if (train_args.local_rank == 0) and valid_dataset:
                    logger.info(f"{dataset_name[start:end]}-valid_dataset")
                    logger.info(part_dataset)
                valid_dataset_dict[dataset_name[start:end]] = part_dataset
            valid_dataset = valid_dataset_dict
        else:
            if (train_args.local_rank == 0) and valid_dataset:
                logger.info("valid_dataset")
                logger.info(valid_dataset)

    test_dataset = None
    if train_args.do_predict:
        test_dataset = collect_dataset(train_args.test_dataset_prefix)
        if (train_args.local_rank == 0) and test_dataset:
            logger.info("test_dataset")
            logger.info(test_dataset)

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

    check_wandb = ("wandb" in train_args.report_to) and (train_args.local_rank == 0)
    if is_wandb_available() and check_wandb:
        import wandb

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            group=os.getenv("WANDB_RUN_GROUP"),
            name=train_args.run_name,
            save_code=True,
        )
        GLOBAL_LOGGER = wandb

    main(train_args)

    if GLOBAL_LOGGER:
        GLOBAL_LOGGER.finish()
