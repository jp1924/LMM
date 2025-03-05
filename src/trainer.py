import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, RandomSampler, Sampler
from trl.trainer.utils import DataCollatorForCompletionOnlyLM

from transformers import Trainer, TrainingArguments
from transformers import logging as hf_logging
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length, seed_worker
from transformers.utils import is_datasets_available


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


class PackingImageCollatorForCompletionOnlyLM(DataCollatorForCompletionOnlyLM):
    def __init__(
        self,
        args: TrainingArguments,
        dtype: torch.dtype,
        sample_dataset: Optional[Dataset] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dtype = dtype
        self.args = args

        if sample_dataset and args.is_world_process_zero:
            formated_instruct = self.tokenizer.decode(sample_dataset[0]["input_ids"], skip_special_tokens=False)
            logger.info(f"formated_instruct: {formated_instruct}")
            logger.info(
                [(self.tokenizer.convert_ids_to_tokens(ids), ids) for ids in sample_dataset[0]["input_ids"][0]]
            )

            if args.response_template is not None:
                response_template = self.tokenizer.decode(args.response_template, skip_special_tokens=False)
                logger.info(f"response_template: {response_template}")
                if response_template not in formated_instruct:
                    raise ValueError("이거 response_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈")
            else:
                raise logger.error("response_template이 없음. 다시 설정하셈.")

            if args.instruction_template is not None:
                instruction_template = self.tokenizer.decode(args.instruction_template, skip_special_tokens=False)
                logger.info(f"instruction_template: {instruction_template}")
                if instruction_template not in formated_instruct:
                    raise ValueError(
                        "이거 instruction_template이 formated_instruct에 포함되어 있지 않음. 다시 설정하셈"
                    )
            else:
                logger.warning("instruction_template이 없음. 근데 애러는 발생하지 않고 그냥 패스함.")

        sample_check = self([sample_dataset[0]])
        if self.args.is_world_process_zero:
            sample_check["labels"] = sample_check["labels"][sample_check["labels"] != -100].tolist()
            check_labels = [self.tokenizer.convert_ids_to_tokens(token) for token in sample_check["labels"]]
            check_labels = ", ".join(check_labels)
            logger.info(f"collator_label: [-100,  ..., -100, {check_labels}]")

        if (
            self.tokenizer.bos_token_id is not None
            and self.tokenizer.bos_token_id not in sample_check["input_ids"].tolist()[0]
        ):
            raise ValueError("BOS token이 없다. 이거 다시 전처리 해라.")

        if self.tokenizer.eos_token_id not in sample_check["input_ids"].tolist()[0]:
            raise ValueError("EOS token이 없다. 이거 다시 전처리 해라.")

    def _create_attention_mask(self, input_length_ls):
        total_length = sum(input_length_ls)
        attention_mask = torch.full((1, 1, total_length, total_length), torch.finfo(self.dtype).min)

        start_idx, end_idx = 0, 0
        for length in input_length_ls:
            end_idx += length
            one_tensor = torch.ones((length, length), dtype=torch.float32)
            mask = torch.tril(one_tensor, diagonal=0).to(dtype=torch.bool)
            attention_mask[0, 0, start_idx:end_idx, start_idx:end_idx][mask] = 0
            start_idx = end_idx

        return attention_mask

    def _process_features(self, features_ls: List[Union[List[int], Any, Dict[str, Any]]]):
        input_ids_ls, labels_ls, position_ids_ls, input_length_ls, pixel_values_ls = [], [], [], [], []
        for features in features_ls:
            batch = super().torch_call([{"input_ids": features["input_ids"]}])
            input_ids, labels = batch.input_ids[0], batch.labels[0]
            length = len(input_ids)

            labels_ls.append(labels)
            input_ids_ls.append(input_ids)
            input_length_ls.append(length)
            position_ids_ls.append(torch.arange(length))

            if features["pixel_values"] is not None:
                pixel_values_ls.append(features["pixel_values"])

        return (input_ids_ls, labels_ls, position_ids_ls, input_length_ls, pixel_values_ls)

    def torch_call(self, features_ls):
        if isinstance(features_ls[0], dict):
            batch = self._process_features(features_ls)
            input_ids_ls, labels_ls, position_ids_ls, input_length_ls, pixel_values_ls = batch
        elif isinstance(features_ls[0], list):
            input_ids_ls, labels_ls, position_ids_ls, input_length_ls, pixel_values_ls = [], [], [], [], []
            for packing_ls in features_ls:
                ids, labels, positions, lengths, pixel_values = self._process_features(packing_ls)

                input_ids_ls.extend(ids)
                labels_ls.extend(labels)
                position_ids_ls.extend(positions)
                input_length_ls.extend(lengths)
                pixel_values_ls.extend(pixel_values)

        attention_mask = self._create_attention_mask(input_length_ls)

        batch = {
            "labels": torch.concat(labels_ls)[None],
            "input_ids": torch.concat(input_ids_ls)[None],
            "position_ids": torch.concat(position_ids_ls)[None],
            "attention_mask": attention_mask,
        }

        if pixel_values_ls:
            batch["pixel_values"] = torch.stack(pixel_values_ls, dim=0)

        return batch


class PackingSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        lengths: List[int],
        max_seq_len: int,
        max_seq_per_pack: int,
        do_shuffle: bool = False,
    ):
        self.dataset = dataset

        self.packing_strategies = self._get_packing_strategies(
            lengths=lengths,
            max_seq_len=max_seq_len,
            max_seq_per_pack=max_seq_per_pack,
        )

        self.do_shuffle = do_shuffle
        self.lengths = lengths

        self.packing_sample_ls = self._transform_length_to_indices(
            strategies_per_length=self.packing_strategies,
            lengths=lengths,
        )

    def _get_packing_strategies(
        self,
        lengths: List[int],
        max_seq_len: int,
        max_seq_per_pack: int,
    ) -> dict:
        def add_pack(
            pack: List[int],
            count: int,
            tmp: defaultdict,
            final: defaultdict,
            limit: int,
            offset: int,
        ) -> None:
            if len(pack) == limit or offset == 0:
                final[offset].append((count, pack))
            else:
                tmp[offset].append((count, pack))

        seq_lens, counts = np.unique(lengths, return_counts=True)
        histogram = np.zeros(max_seq_len, dtype=np.int64)
        histogram[seq_lens - 1] = counts

        reversed_histogram = np.flip(histogram)

        tmp_strategies_per_length = defaultdict(list)
        strategies_per_length = defaultdict(list)

        for i in range(max_seq_len):
            n_sequences_to_bin = reversed_histogram[i]
            length_to_bin = max_seq_len - i
            offset = i + 1  # largest possible offset
            while n_sequences_to_bin > 0:
                if (length_to_bin + offset) in tmp_strategies_per_length:
                    # extract shortest pack that will get modified
                    n_sequences_to_pack, pack = tmp_strategies_per_length[length_to_bin + offset].pop()
                    new_pack = pack + [length_to_bin]
                    count = min(n_sequences_to_pack, n_sequences_to_bin)
                    if n_sequences_to_pack > n_sequences_to_bin:
                        # old pack gets reduced
                        n_sequences_to_pack -= n_sequences_to_bin
                        tmp_strategies_per_length[length_to_bin + offset].append((n_sequences_to_pack, pack))
                        n_sequences_to_bin = 0
                    else:
                        n_sequences_to_bin -= n_sequences_to_pack
                    add_pack(
                        new_pack, count, tmp_strategies_per_length, strategies_per_length, max_seq_per_pack, offset
                    )
                    # clean up to speed up main key search
                    if not tmp_strategies_per_length[length_to_bin + offset]:
                        tmp_strategies_per_length.pop(length_to_bin + offset)
                else:
                    offset -= 1
                # Does not fit anywhere. Create new pack.
                if offset < 0:
                    add_pack(
                        [length_to_bin],
                        n_sequences_to_bin,
                        tmp_strategies_per_length,
                        strategies_per_length,
                        max_seq_per_pack,
                        i,
                    )
                    n_sequences_to_bin = 0
        # merge all strategies
        for key in tmp_strategies_per_length:
            strategies_per_length[key].extend(tmp_strategies_per_length[key])

        return strategies_per_length

    def _transform_length_to_indices(self, strategies_per_length: dict, lengths: List[int]) -> List[List[int]]:
        length_to_indices = {}
        length_array = np.array(lengths)
        unique_lengths = np.unique(length_array).tolist()

        for length in unique_lengths:
            dataset_idx_ls = np.where(length_array == length)[0].tolist()
            if self.do_shuffle:
                random.shuffle(dataset_idx_ls)
            length_to_indices[length] = dataset_idx_ls

        pack_strategies_ls = [
            pack
            for strategies in strategies_per_length.values()
            for strategies_num, pack_strategies in strategies
            for pack in ([pack_strategies] * strategies_num)
        ]

        packing_sample_ls = list()
        for pack_strategies in pack_strategies_ls:
            pack_size = len(pack_strategies)
            strategie_position = 0

            dataset_idx_ls = list()
            while strategie_position + 1 <= pack_size:
                length = pack_strategies[strategie_position]
                pack_length_ls = length_to_indices[length]
                dataset_idx_ls.append(pack_length_ls.pop())
                length_to_indices[length] = pack_length_ls
                strategie_position += 1

            packing_sample_ls.append(dataset_idx_ls)

        if self.do_shuffle:
            random.shuffle(packing_sample_ls)

        return packing_sample_ls

    def __iter__(self):
        if self.do_shuffle:
            packing_sample_ls = self._transform_length_to_indices(
                strategies_per_length=self.packing_strategies,
                lengths=self.lengths,
            )
        else:
            packing_sample_ls = self.packing_sample_ls

        return iter(packing_sample_ls)

    def __len__(self):
        return len(self.packing_sample_ls)


class PackingTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        def __packing_getitems__(train_dataset, keys: List[List[int]]) -> List:
            """Can be used to get a batch using a list of integers indices."""

            return_ls = list()
            for key in keys:
                batch = train_dataset.__getitem__(key)
                n_examples = len(batch[next(iter(batch))])

                return_ls.append([{col: array[i] for col, array in batch.items()} for i in range(n_examples)])
            return return_ls

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        # NOTE: packing을 사용할 경우 packing에 알맞은 getitems를 사용하도록 합니다.
        if self.args.do_packing:
            # 래핑된 함수를 정의하여 self를 전달할 수 있도록 합니다.
            def getitems_wrapper(keys):
                return __packing_getitems__(train_dataset, keys)

            setattr(self.train_dataset, "__getitems__", getitems_wrapper)

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_length and self.args.do_packing:
            raise ValueError("group_by_length and do_packing cannot be used together.")

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0] if self.processing_class is not None else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        elif self.args.do_packing:
            if is_datasets_available() and isinstance(self.train_dataset, Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None

            return PackingSampler(
                dataset=self.train_dataset,
                lengths=lengths,
                max_seq_len=self.args.data_max_length,
                max_seq_per_pack=self.args.packing_max_elem,
                do_shuffle=self.args.packing_shuffle,
            )

        else:
            return RandomSampler(self.train_dataset)

    def create_optimizer(self):
        if self.model.config.model_type != "llava_onevision":
            return super().create_optimizer()
        decay_parameters = self.get_decay_parameter_names(self.model)

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.multi_modal_projector.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.multi_modal_projector.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.language_model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.language_model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.vision_tower.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.vision_learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.vision_tower.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
                "lr": self.args.vision_learning_rate,
            },
        ]

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, self.model)

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


__all__ = ["PackingImageCollatorForCompletionOnlyLM", "PackingSampler", "PackingTrainer"]
