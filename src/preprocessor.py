import json
import time
from typing import Callable, Optional, Tuple

from datasets import Dataset, concatenate_datasets, load_dataset

from transformers import ProcessorMixin, TrainingArguments
from transformers import logging as hf_logging


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def llava_stage1_preprocessor(example, processor: ProcessorMixin, args: TrainingArguments):
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

    preprocess_finish_ls = list()
    for idx, conversations in enumerate(example["conversations"]):
        for chat in conversations:
            content = json.loads(chat["content"])
            if isinstance(content, list):
                for part in content:
                    if part["type"] != "text":
                        continue

                    part["text"] = (
                        str(part["text"]) if isinstance(part["text"], (int, float)) else part["text"].strip()
                    )

            content = str(content) if isinstance(content, (int, float)) else content
            chat["content"] = content

        text = processor.apply_chat_template(conversations, tokenize=False)
        image = example["image"][idx] if "image" in example else None
        image = [i.convert("RGB") for i in image] if isinstance(image, list) else image.convert("RGB")
        num_images = len(image) if isinstance(image, list) else 1

        if image and text.count(processor.image_token) != num_images:
            logger.info(
                f"text: {text}\n"
                f"image: {image}\n"
                f"input_ids: {text}\n"
                "image and (config.image_token_id not in input_ids) 필터링 됨.\n"
            )
            continue
        elif (image is None) and (processor.image_token in text):
            logger.info(
                f"text: {text}\n"
                f"image: {image}\n"
                f"input_ids: {text}\n"
                "(image is None) and (config.image_token_id in input_ids) 필터링 됨."
            )
            continue

        outputs = processor(text=text, images=image, return_tensors="np")

        preprocess_finish_ls.append(
            {
                "pixel_values": outputs["pixel_values"][0] if image else None,
                "input_ids": outputs["input_ids"][0],
                args.length_column_name: outputs["input_ids"][0].shape[0],
            }
        )

    return_dict = dict()
    for res in preprocess_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def llava_stage2_preprocessor(example, processor: ProcessorMixin, args: TrainingArguments):
    preprocess_finish_ls = list()
    for idx, conversations in enumerate(example["conversations"]):
        for chat in conversations:
            content = json.loads(chat["content"])
            content = str(content) if isinstance(content, (int, float)) else content
            chat["content"] = content

        image = example["image"][idx] if "image" in example else None
        image = [i.convert("RGB") for i in image] if isinstance(image, list) else image.convert("RGB")
        num_images = len(image) if isinstance(image, list) else 1

        text = processor.apply_chat_template(conversations, tokenize=False)

        if image and text.count(processor.image_token) != num_images:
            logger.info(
                f"text: {text}\n"
                f"image: {image}\n"
                f"input_ids: {text}\n"
                "image and (config.image_token_id not in input_ids) 필터링 됨.\n"
            )
            continue
        elif (image is None) and (processor.image_token in text):
            logger.info(
                f"text: {text}\n"
                f"image: {image}\n"
                f"input_ids: {text}\n"
                "(image is None) and (config.image_token_id in input_ids) 필터링 됨."
            )
            continue

        outputs = processor(text=text, images=image, return_tensors="np")

        preprocess_finish_ls.append(
            {
                "pixel_values": outputs["pixel_values"][0] if image else None,
                "input_ids": outputs["input_ids"][0],
                args.length_column_name: outputs["input_ids"][0].shape[0],
            }
        )

    return_dict = dict()
    for res in preprocess_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def llava_next_stage1_5_preprocessor(example, processor: ProcessorMixin, args: TrainingArguments):
    preprocess_finish_ls = list()
    for image, conversations in zip(example["image"], example["conversations"]):
        for chat in conversations:
            content = json.loads(chat["content"])
            content = str(content) if isinstance(content, (int, float)) else content
            chat["content"] = content

        text = processor.apply_chat_template(conversations, tokenize=False)
        image = [i.convert("RGB") for i in image] if isinstance(image, list) else image.convert("RGB")
        num_images = len(image) if isinstance(image, list) else 1

        if image and text.count(processor.image_token) != num_images:
            logger.info(
                f"text: {text}\n"
                f"image: {image}\n"
                f"input_ids: {text}\n"
                "image and (config.image_token_id not in input_ids) 필터링 됨.\n"
            )
            continue
        elif (image is None) and (processor.image_token in text):
            logger.info(
                f"text: {text}\n"
                f"image: {image}\n"
                f"input_ids: {text}\n"
                "(image is None) and (config.image_token_id in input_ids) 필터링 됨."
            )
            continue

        outputs = processor(text=text, images=image, return_tensors="np")

        preprocess_finish_ls.append(
            {
                "pixel_values": outputs["pixel_values"][0] if image else None,
                "input_ids": outputs["input_ids"][0],
                "image_sizes": outputs["image_sizes"][0],
                args.length_column_name: outputs["input_ids"][0].shape[0],
            }
        )

    return_dict = dict()
    for res in preprocess_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


def llava_next_stage2_preprocessor(example, processor: ProcessorMixin, args: TrainingArguments):
    preprocess_finish_ls = list()
    for image, conversations in zip(example["image"], example["conversations"]):
        for chat in conversations:
            content = json.loads(chat["content"])
            chat["content"] = content

        text = processor.apply_chat_template(conversations, tokenize=False)
        image = [i.convert("RGB") for i in image] if isinstance(image, list) else image.convert("RGB")
        num_images = len(image) if isinstance(image, list) else 1

        if image and text.count(processor.image_token) != num_images:
            logger.info(
                f"text: {text}\n"
                f"image: {image}\n"
                f"input_ids: {text}\n"
                "image and (config.image_token_id not in input_ids) 필터링 됨.\n"
            )
            continue
        elif (image is None) and (processor.image_token in text):
            logger.info(
                f"text: {text}\n"
                f"image: {image}\n"
                f"input_ids: {text}\n"
                "(image is None) and (config.image_token_id in input_ids) 필터링 됨."
            )
            continue

        outputs = processor(text=text, images=image, return_tensors="np")

        preprocess_finish_ls.append(
            {
                "pixel_values": outputs["pixel_values"][0] if image else None,
                "input_ids": outputs["input_ids"][0],
                "image_sizes": outputs["image_sizes"][0],
                args.length_column_name: outputs["input_ids"][0].shape[0],
            }
        )

    return_dict = dict()
    for res in preprocess_finish_ls:
        for key, value in res.items():
            return_dict.setdefault(key, []).append(value)

    return return_dict


PROCESSOR_REGISTRY = {
    "llava_stage-1.0": llava_stage1_preprocessor,
    "llava_stage-2.0": llava_stage2_preprocessor,
    "llava_next_stage-1.5": llava_next_stage1_5_preprocessor,
    "llava_next_stage-2.0": llava_next_stage2_preprocessor,
}


def processing_datasets(
    func: Callable, train_args: TrainingArguments, processor: ProcessorMixin
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
