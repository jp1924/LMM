import json

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
