from typing import Tuple

import torch

from transformers import (
    AddedToken,
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPConfig,
    CLIPModel,
    CLIPProcessor,
    CLIPVisionModel,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaNextImageProcessor,
    LlavaNextProcessor,
    LlavaProcessor,
)


IMG_TOKEN = "<|image|>"
CHAT_TEMPLATE = """{% for message in messages %}{% if message.role == 'user' %}{{ '### User:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ img_token }}{% elif content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'assistant' %}{{ '### Assistant:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ img_token }}{% elif content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{% endif %}{% endfor %}{{ eos_token }}"""


def get_vision_processor_and_model(vision_model_name_or_path) -> Tuple[AutoModel, AutoImageProcessor, AutoConfig]:
    vision_tower = AutoModel.from_pretrained(vision_model_name_or_path)
    vision_config = AutoConfig.from_pretrained(vision_model_name_or_path)
    processor = AutoImageProcessor.from_pretrained(vision_model_name_or_path)

    if isinstance(vision_config, CLIPConfig):
        vision_config = vision_config.vision_config

    if isinstance(processor, CLIPProcessor):
        image_processor = processor.image_processor

        if "shortest_edge" in image_processor.size:
            image_processor.size = image_processor.size["shortest_edge"]

    if isinstance(vision_tower, CLIPModel):
        vision_tower = CLIPVisionModel.from_pretrained(vision_model_name_or_path, config=vision_config)

    return (vision_tower, vision_config, processor)


def get_language_tokenizer_and_model(language_model_name_or_path) -> Tuple[AutoModel, AutoTokenizer, AutoConfig]:
    language_model = AutoModelForCausalLM.from_pretrained(language_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        language_model_name_or_path,
        padding_side="left",
        chat_template=CHAT_TEMPLATE,
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_tokens(AddedToken(IMG_TOKEN, special=True, normalized=False), special_tokens=True)

    new_vocab_size = len(tokenizer.get_vocab())

    embedding = language_model.resize_token_embeddings(new_vocab_size)
    language_model.set_input_embeddings(embedding)
    language_config = AutoConfig.from_pretrained(
        language_model_name_or_path,
        vocab_size=new_vocab_size,
        padding_idx=tokenizer.pad_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        unk_token_id=tokenizer.unk_token_id,
    )

    return (language_model, language_config, tokenizer)


def main() -> None:
    vision_model_name_or_path = "Bingsu/clip-vit-large-patch14-ko"
    language_model_name_or_path = "beomi/llama-2-koen-13b"

    vision_tower, vision_config, image_processor = get_vision_processor_and_model(vision_model_name_or_path)
    language_model, language_config, tokenizer = get_language_tokenizer_and_model(language_model_name_or_path)

    # from mplugDocOwl
    image_grid_pinpoints = (
        torch.tensor(
            [
                (1, 1),
                (1, 2),
                (2, 1),
                (1, 3),
                (3, 1),
                (2, 2),
                (1, 4),
                (4, 1),
                (1, 5),
                (5, 1),
                (1, 6),
                (6, 1),
                (2, 3),
                (3, 2),
                (1, 7),
                (7, 1),
                (4, 2),
                (2, 4),
                (1, 8),
                (8, 1),
                (3, 3),
                (1, 9),
                (9, 1),
            ]
        )
        * vision_config.image_size
    ).tolist()

    setattr(vision_config, "_name_or_path", vision_model_name_or_path)
    setattr(image_processor, "image_grid_pinpoints", image_grid_pinpoints)
    setattr(image_processor, "do_pad", True)

    # stage-1 학습을 위해, vision model의 image_prcessor로 저장함, 반대로 llava next image processor로 불러오는 것도 가능함.
    # llava_image_processor = LlavaNextImageProcessor(
    #     do_resize=image_processor.do_resize,
    #     size=image_processor.size,
    #     resample=image_processor.resample,
    #     do_center_crop=image_processor.do_center_crop,
    #     do_rescale=image_processor.do_rescale,
    #     rescale_factor=image_processor.rescale_factor,
    #     do_normalize=image_processor.do_normalize,
    #     image_mean=image_processor.image_mean,
    #     image_std=image_processor.image_std,
    #     do_convert_rgb=image_processor.do_convert_rgb,
    #     image_grid_pinpoints=image_grid_pinpoints,
    #     do_pad=True,
    # )
    processor = LlavaProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        chat_template=tokenizer.chat_template,
    )

    config = LlavaConfig(
        vision_config=vision_config,
        text_config=language_config,
        image_grid_pinpoints=image_grid_pinpoints,
        image_token_index=tokenizer.convert_tokens_to_ids(IMG_TOKEN),
    )
    model = LlavaForConditionalGeneration(config)

    # TODO: 나중에 set_decoder로 바꿀 것.
    model.vision_tower = vision_tower
    model.language_model = language_model

    hub_name = "Llama-ClipLarge-Llava"
    output_dir = f"/root/output_dir/{hub_name}"
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    max_retry = 10
    for retries in range(max_retry):
        try:
            model.push_to_hub(hub_name, private=True)
            processor.push_to_hub(hub_name, private=True)
        except BaseException as e:
            print(f"해당 애러가 {retries}시에 발생: {e}")
    else:
        exit("모델이 정상적으로 업로드 되질 않았음. 프로그램을 종료함.")


if "__main__" in __name__:
    try:
        processor = LlavaProcessor.from_pretrained("/root/output_dir/Llama-ClipLarge-Llava")
        model = LlavaForConditionalGeneration.from_pretrained("/root/output_dir/Llama-ClipLarge-Llava")
    except BaseException as e:
        main()
    max_retry = 10
    for retries in range(max_retry):
        try:
            model.push_to_hub("Llama-ClipLarge-Llava", private=True)
            processor.push_to_hub("Llama-ClipLarge-Llava", private=True)
        except BaseException as e:
            print(f"해당 애러가 {retries}시에 발생: {e}")
    else:
        exit("모델이 정상적으로 업로드 되질 않았음. 프로그램을 종료함.")