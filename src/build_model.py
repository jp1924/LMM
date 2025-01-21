import json
import shutil
from pathlib import Path
from typing import Tuple

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlavaConfig,
    LlavaForConditionalGeneration,
    LlavaProcessor,
    PreTrainedModel,
    PreTrainedTokenizer,
    ProcessorMixin,
)


IMG_TOKEN = "<img>"
CHAT_TEMPLATE = "{% if not add_last_empty_assistant is defined %}{% set add_last_empty_assistant = false %}{% endif %}{% if not include_last_assistant is defined %}{% set include_last_assistant = true %}{% endif %}{% for message in messages %}{{ sot_token }}{% if message.role == 'user' %}{{ '### User:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ img_token }}{% elif content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'system' %}{{ '### System:\n' }}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'image' %}{{ img_token }}{% elif content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{{ '\n\n' }}{% elif message.role == 'assistant' %}{{ '### Assistant:\n' }}{% if not loop.last or include_last_assistant %}{% if message.content is not string %}{% for content in message.content %}{% if content.type == 'text' %}{{ content.text }}{% else %}{# Do nothing #}{% endif %}{% endfor %}{% else %}{{ message.content }}{% endif %}{% endif %}{% else %}{# Do nothing #}{% endif %}{{ eot_token }}{% endfor %}{% if not include_last_assistant %}{# Do nothing #}{% elif not add_last_empty_assistant %}{{ eos_token }}{% elif add_last_empty_assistant %}{{ '### Assistant:\n' }}{% else %}{# Do nothing #}{% endif %}"


def get_vision(vision_model_name_or_path) -> Tuple[AutoModel, AutoConfig, AutoImageProcessor]:
    config = AutoConfig.from_pretrained(vision_model_name_or_path)
    processor = AutoImageProcessor.from_pretrained(vision_model_name_or_path)

    if hasattr(config, "vision_config"):
        config = getattr(config, "vision_config")

    model = AutoModel.from_config(config)
    model = model.from_pretrained(vision_model_name_or_path)

    return (model, config, processor)


def get_language(language_model_name_or_path) -> Tuple[AutoModel, AutoConfig, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(language_model_name_or_path)
    config = AutoConfig.from_pretrained(language_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(language_model_name_or_path, padding_side="left")
    if tokenizer.pad_token == tokenizer.eos_token:
        print(f"tokenizer의 pad_token이 {tokenizer.pad_token}과 같이 되어 있어서 {tokenizer.unk_token}으로 변경함.")
        tokenizer.pad_token = tokenizer.unk_token

    return (model, config, tokenizer)


def insert_img_token_to_gemma_tokenizer(tokenizer: PreTrainedTokenizer, img_token) -> Tuple[PreTrainedTokenizer, int]:
    img_token_dict = {
        "content": img_token,
        "lstrip": False,
        "normalized": False,
        "rstrip": False,
        "single_word": False,
        "special": True,
    }

    unused_prefix = "unused"
    unsed_token_ls = [vocab_row for vocab_row in tokenizer.get_added_vocab().items() if unused_prefix in vocab_row[0]]
    unused_ls = sorted(unsed_token_ls, key=lambda vocab_row: vocab_row[1])

    if not unused_ls:
        raise ValueError("unused token이 존재하지 않음.")

    unused_token, unused_idx = unused_ls[0]

    model_dir = Path(tokenizer.vocab_file).parent
    tokenizer_config = json.loads(model_dir.joinpath("tokenizer_config.json").read_text())
    special_tokens_map = json.loads(model_dir.joinpath("special_tokens_map.json").read_text())
    tokenizer_raw_file = json.loads(model_dir.joinpath("tokenizer.json").read_text())

    tokenizer_config["added_tokens_decoder"][str(unused_idx)] = img_token_dict
    special_tokens_map["img_token"] = {k: v for k, v in img_token_dict.items() if k != "special"}

    tokenizer_raw_file["added_tokens"][unused_idx] = {"id": unused_idx, **img_token_dict}
    tokenizer_raw_file["model"]["vocab"].pop(unused_token)
    tokenizer_raw_file["model"]["vocab"][img_token] = unused_idx

    save_path = model_dir.joinpath("multi_modal_tokenizer")
    save_path.mkdir(exist_ok=True)

    save_path.joinpath("tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2, ensure_ascii=False))
    save_path.joinpath("special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2, ensure_ascii=False)
    )
    save_path.joinpath("tokenizer.json").write_text(json.dumps(tokenizer_raw_file, indent=2, ensure_ascii=False))

    mm_tokenizer = AutoTokenizer.from_pretrained(save_path.as_posix())

    # NOTE: Remove the saved files
    shutil.rmtree(save_path)

    return mm_tokenizer, unused_idx


def upload_to_hub(model: PreTrainedModel, processor: ProcessorMixin, hub_name: str, upload_retry: int = 10):
    for retries in range(upload_retry):
        try:
            model.push_to_hub(hub_name, private=True)
            processor.push_to_hub(hub_name, private=True)
        except BaseException as e:
            print(f"해당 애러가 {retries}시에 발생: {e}")
    else:
        exit("모델이 정상적으로 업로드 되질 않았음. 프로그램을 종료함.")


def main(
    language_model_name_or_path: str,
    vision_model_name_or_path: str,
    output_dir: str,
    img_token: str = IMG_TOKEN,
    chat_template: str = CHAT_TEMPLATE,
    chat_template_forced: bool = False,
    push_to_hub: bool = False,
    upload_retry: int = 10,
):
    vision_model, vision_config, vision_processor = get_vision(vision_model_name_or_path)
    language_model, language_config, language_tokenizer = get_language(language_model_name_or_path)

    if "gemma" == language_config.model_type:
        language_tokenizer, image_token_index = insert_img_token_to_gemma_tokenizer(language_tokenizer, img_token)
    elif "gemma2" == language_config.model_type:
        language_tokenizer, image_token_index = insert_img_token_to_gemma_tokenizer(language_tokenizer, img_token)
    else:
        raise ValueError("지원하는 모델이 아님.")

    if chat_template_forced or language_tokenizer.chat_template is None:
        language_tokenizer.chat_template = chat_template
        print(f"tokenizer의 chat_template을 {chat_template}으로 변경")
    else:
        print(
            f"chat_template이 {language_tokenizer.chat_template}이라 따로 변경하지 않음. 변경하고 싶으면면 chat_template_forced를 True로 변경"
        )

    image_size, patch_size = vision_config.image_size, vision_config.patch_size
    # tokens_per_dim = image_size // patch_size
    # num_tokens = tokens_per_dim * tokens_per_dim

    config = LlavaConfig(
        vision_config=vision_config,
        text_config=language_config,
        image_seq_length=image_size,
        image_token_index=image_token_index,
    )
    processor = LlavaProcessor(
        tokenizer=language_tokenizer,
        image_processor=vision_processor,
        image_token=img_token,
        vision_feature_select_strategy=config.vision_feature_select_strategy,
        patch_size=patch_size,
        chat_template=language_tokenizer.chat_template,
    )

    model = LlavaForConditionalGeneration(config)

    model.vision_tower = vision_model
    model.language_model = language_model

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    LlavaForConditionalGeneration.from_pretrained(output_dir)
    LlavaProcessor.from_pretrained(output_dir)

    if push_to_hub:
        hub_name = Path(output_dir).name
        upload_to_hub(model, processor, hub_name, upload_retry)


if "__main__" in __name__:
    vision_model_name_or_path = "google/siglip-so400m-patch14-384"
    language_model_name_or_path = "google/gemma-2-9b"

    name = "KoLLaVa9b-patch14-384"
    output_dir = f"/scratch/slurm-user18-42maru/home/jp/output_dir/{name}"
    model, processor = main(
        language_model_name_or_path,
        vision_model_name_or_path,
        output_dir,
    )
