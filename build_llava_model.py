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
    LlavaProcessor,
)


IMG_TOKEN = "<|image|>"
CHAT_TEMPLATE = """{% if not add_last_empty_assistant is defined %}{% set add_last_empty_assistant = false %}{% endif %}{% for message in messages %}{{ sot_token }}{% if message.role == 'user' %}{{ '### User:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ img_token }}{% elif content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'system' %}{{ '### System:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ img_token }}{% elif content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'assistant' %}{{ '### Assistant:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% if not loop.last %}{{ '\n\n' }}{% endif %}{% else %}{{ message.content }}{% if not loop.last %}{{ '\n\n' }}{% endif %}{% endif %}{% else %}{# Do nothing #}{% endif %}{{ eot_token }}{% endfor %}{% if not add_last_empty_assistant %}{{ eos_token }}{% elif add_last_empty_assistant %}{{ '### Assistant:\n' }}{% else %}{# Do nothing #}{% endif %}"""


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


def main(
    language_model_name_or_path: str,
    vision_model_name_or_path: str,
    output_dir: str,
):
    vision_tower, vision_config, image_processor = get_vision_processor_and_model(vision_model_name_or_path)
    language_model, language_config, tokenizer = get_language_tokenizer_and_model(language_model_name_or_path)

    # NOTE: LLaVA-NEXT Abalation 논문을 보고 수정함.
    image_grid_pinpoints = (
        torch.tensor(
            [
                (1, 1),
                (1, 2),
                (2, 1),
                (2, 2),
                (1, 3),
                (3, 1),
                (2, 3),
                (3, 2),
                (3, 3),
                (1, 4),
                (4, 1),
                (2, 4),
                (4, 2),
                (3, 4),
                (4, 3),
                (4, 4),
            ]
        )
        * vision_config.image_size
    ).tolist()

    setattr(vision_config, "_name_or_path", vision_model_name_or_path)
    setattr(image_processor, "image_grid_pinpoints", image_grid_pinpoints)
    setattr(image_processor, "do_pad", True)

    config = LlavaConfig(
        vision_config=vision_config,
        image_seq_length=vision_config.image_size,
        text_config=language_config,
        image_grid_pinpoints=image_grid_pinpoints,
        image_token_index=tokenizer.convert_tokens_to_ids(IMG_TOKEN),
    )
    processor = LlavaProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token=IMG_TOKEN,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        patch_size=config.vision_config.patch_size,
        chat_template=tokenizer.chat_template,
    )

    model = LlavaForConditionalGeneration(config)

    # TODO: 나중에 set_decoder로 바꿀 것.
    model.vision_tower = vision_tower
    model.language_model = language_model

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    return (model, processor)


if "__main__" in __name__:
    vision_model_name_or_path = "Bingsu/clip-vit-large-patch14-ko"
    language_model_name_or_path = "beomi/llama-2-ko-7b"

    hub_name = "KoLLaVA-7b"
    output_dir = f"/root/output_dir/llava/{hub_name}"
    try:
        processor = LlavaProcessor.from_pretrained(output_dir)
        model = LlavaForConditionalGeneration.from_pretrained(output_dir)
    except BaseException as e:  # noqa: F841
        model, processor = main(
            language_model_name_or_path,
            vision_model_name_or_path,
            output_dir,
        )

    max_retry = 10
    for retries in range(max_retry):
        try:
            model.push_to_hub(hub_name, private=True)
            processor.push_to_hub(hub_name, private=True)
        except BaseException as e:
            print(f"해당 애러가 {retries}시에 발생: {e}")
    else:
        exit("모델이 정상적으로 업로드 되질 않았음. 프로그램을 종료함.")
