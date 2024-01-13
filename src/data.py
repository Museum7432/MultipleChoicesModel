import lightning as L

from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
import numpy as np
import torch
import random
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split


def randomsent(length=20):
    return "".join(random.choice("abcdefghijklmnopqrstuvwxyz ") for i in range(length))


class SemevalDataset(Dataset):
    def __init__(
        self,
        raw_data,
        encoder_name,
        promt_style=True,
        include_label=True,
        shuffle_choices=False,
        reduce_choices=False,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        self.include_label = include_label

        self.promt_style = promt_style

        self.shuffle_choices = shuffle_choices

        if self.promt_style:
            # to match with flan-t5 promt style
            self.choice_indicators = [
                " (A) ",
                " (B) ",
                " (C) ",
                " (D) ",
                " (E) ",
                " (F) ",
                " (G) ",
            ]
            # we should focus on the token that represents the character 'A' in " (A) "
            self.main_indicator_char_offset = [
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        else:
            self.choice_indicator = self.tokenizer.eos_token

        self.reduce_choices = reduce_choices

        self.entries = [self._process_raw(item) for item in raw_data]

    def _process_raw(self, item):
        # assert item["choice_list"][item["label"]] == item["answer"]

        # if item["choice_list"][item["label"]] != item["answer"]:
        #     print(item)

        if self.promt_style:
            for ci in self.choice_indicators:
                assert ci not in item["question"]

                for c in item["choice_list"]:
                    assert ci not in c
        else:
            assert self.choice_indicator not in item["question"]

            for c in item["choice_list"]:
                assert self.choice_indicator not in c

        if self.reduce_choices:
            assert item["choice_list"][-1] == "None of above."

        # remove the newline char
        item["question"] = item["question"].replace("\n", "")

        item["choice_list"] = [s.replace("\n", "") for s in item["choice_list"]]

        if self.include_label:
            return {
                "question": item["question"],
                "choice_list": item["choice_list"],
                "label": item["label"],
            }

        return {
            "question": item["question"],
            "choice_list": item["choice_list"],
        }

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]

        if self.include_label:
            label = item["label"]

        choice_list = item["choice_list"].copy()

        # assume that the last item of choice_list is always "None of above."
        # randomly remove 0, 1, or 2 of the other choices into a random string or remove that option entirely

        if self.reduce_choices:
            if self.include_label:
                answer = choice_list[label]

            if random.random() < 0.66:
                # 66% chance of removing 1 choice or more
                # so 34% chance of having all 4 choices
                indice = random.randint(0, 2)
                choice_list.pop(indice)

                if random.random() < 0.5:
                    # 33% chance of having 3 choices
                    # 33% chance of having 2 choices
                    indice = random.randint(0, 1)
                    choice_list.pop(indice)

            if self.include_label:
                if answer in choice_list:
                    label = choice_list.index(answer)
                else:
                    label = len(choice_list) - 1

            for _ in range(4 - len(choice_list)):
                choice_list.append(randomsent(random.randint(5, 20)))

        if self.shuffle_choices:
            # shuffle the choice list
            tmp = list(enumerate(choice_list))
            random.shuffle(tmp)
            old_indices, choice_list = zip(*tmp)

        if self.promt_style:
            input_text = "Question: ("
            for i in range(len(choice_list)):
                input_text += self.choice_indicators[i][
                    self.main_indicator_char_offset[i]
                ]
                if i != len(choice_list) - 1:
                    input_text += " or "
            input_text += ") "

            input_text += item["question"]
        else:
            input_text = item["question"]

        indicators_char_offset = []

        for idx, choice in enumerate(choice_list):
            indicators_char_offset.append(len(input_text))

            if self.promt_style:
                input_text += self.choice_indicators[idx] + choice

                indicators_char_offset[-1] += self.main_indicator_char_offset[idx]

            else:
                input_text += self.choice_indicator + choice

        if self.promt_style:
            input_text += " Answer: "

        tokenized = self.tokenizer(input_text)

        indicators_token_offset = [
            tokenized.char_to_token(i) for i in indicators_char_offset
        ]

        re = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "last_token_indice": len(tokenized["input_ids"]) - 1,
            "indicators_token_offset": indicators_token_offset,
            "indicators_token_offset_mask": [1] * len(indicators_token_offset),
        }

        if self.include_label:
            if self.shuffle_choices:
                re["label"] = old_indices.index(label)
            else:
                re["label"] = label

        return re


def collate_fn_factory(pad_token_id):
    fields_to_pad = [
        "input_ids",
        "attention_mask",
        "indicators_token_offset",
        "indicators_token_offset_mask",
    ]

    pad_values = [pad_token_id, 0, 0, 0]

    def _pad(arr, pad_value):
        target = max([len(i) for i in arr])
        return [i + [pad_value] * (target - len(i)) for i in arr]

    def collate_fn(items):
        batch = {}

        for f in items[0].keys():
            batch[f] = [i[f] for i in items]

        for f, v in zip(fields_to_pad, pad_values):
            batch[f] = _pad(batch[f], v)

        for f in batch.keys():
            batch[f] = torch.tensor(batch[f])
        return batch

    return collate_fn


class SemevalDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_path: str = None,
        eval_data_path: str = None,
        encoder_name: str = "google/flan-t5-large",
        promt_style: bool = True,
        shuffle_choices: bool = True,
        num_workers: int = 8,
        train_batch_size: int = 2,
        valid_batch_size: int = 4,
        debug: bool = False,
        reduce_choices: bool = False,
    ):
        super().__init__()

        self.train_data_path = train_data_path

        self.eval_data_path = eval_data_path

        self.encoder_name = encoder_name

        self.promt_style = promt_style

        self.shuffle_choices = shuffle_choices

        self.num_workers = num_workers

        self.train_batch_size = train_batch_size

        self.valid_batch_size = valid_batch_size

        self.pad_token_id = None

        self.debug = debug

        self.reduce_choices = reduce_choices

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("SemevalDataModule")
    #     parser.add_argument("--promt_style", action="store_true")
    #     # only for train_dataset
    #     parser.add_argument("--shuffle_choices", action="store_true")
    #     parser.add_argument("--train_data_path", type=str, default=None)
    #     parser.add_argument("--eval_data_path", type=str, default=None)
    #     parser.add_argument("--train_batch_size", type=int, default=2)
    #     parser.add_argument("--valid_batch_size", type=int, default=8)
    #     parser.add_argument("--num_workers", type=int, default=4)
    #     parser.add_argument("--debug", action="store_true")
    #     return parent_parser

    def setup(self, stage: str):
        train_data = list(np.load(self.train_data_path, allow_pickle=True))
        eval_data = list(np.load(self.eval_data_path, allow_pickle=True))

        if self.debug:
            train_data = train_data[:20]
            eval_data = eval_data[:20]

        train_set, valid_set = train_test_split(
            train_data, test_size=0.1, random_state=42
        )

        self.trainDataset = SemevalDataset(
            raw_data=train_set,
            encoder_name=self.encoder_name,
            promt_style=self.promt_style,
            include_label=True,
            shuffle_choices=self.shuffle_choices,
            reduce_choices=self.reduce_choices,
        )

        self.validDataset = SemevalDataset(
            raw_data=valid_set,
            encoder_name=self.encoder_name,
            promt_style=self.promt_style,
            include_label=True,
            shuffle_choices=False,
            reduce_choices=False,
        )

        self.testDataset = SemevalDataset(
            raw_data=eval_data,
            encoder_name=self.encoder_name,
            promt_style=self.promt_style,
            include_label=False,
            shuffle_choices=False,
            reduce_choices=False,
        )

        self.pad_token_id = self.trainDataset.tokenizer.pad_token_id

    def train_dataloader(self):
        return DataLoader(
            self.trainDataset,
            batch_size=self.train_batch_size,
            collate_fn=collate_fn_factory(self.pad_token_id),
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validDataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn_factory(self.pad_token_id),
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.testDataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn_factory(self.pad_token_id),
            shuffle=False,
            num_workers=self.num_workers,
        )
