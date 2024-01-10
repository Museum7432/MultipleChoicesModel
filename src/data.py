import lightning as L

from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer
import numpy as np
import torch
import random
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split


class SemevalDataset(Dataset):
    def __init__(self, raw_data, encoder_name, promt_style=True, include_label=True, shuffle_choices=False):
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

        input_text = item["question"]


        if self.shuffle_choices:
            # shuffle the choice list
            tmp = list(enumerate(item["choice_list"]))
            random.shuffle(tmp)
            old_indices, choice_list = zip(*tmp)
        else:
            choice_list = item["choice_list"]

        indicators_char_offset = []

        for idx, choice in enumerate(choice_list):
            indicators_char_offset.append(len(input_text))

            if self.promt_style:
                input_text += self.choice_indicators[idx] + choice

                indicators_char_offset[-1] += self.main_indicator_char_offset[idx]

            else:
                input_text += self.choice_indicator + choice

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
            "label": old_indices.index(item["label"]),
        }

        if self.include_label:
            re["label"] = old_indices.index(item["label"]),

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

        for f in batch.keys():
            batch[f] = [i[f] for i in items]

        for f, v in zip(fields_to_pad, pad_values):
            batch[f] = _pad(batch[f], v)

        for f in batch.keys():
            batch[f] = torch.tensor(batch[f])
        return batch

    return collate_fn

class SemevalDataModule(L.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        
        self.train_data_path = args.train_data_path

        self.encoder_name = args.encoder_name

        self.promt_style = args.promt_style

        self.shuffle_choices = args.shuffle_choices
        
        self.num_workers = args.num_workers

        self.train_batch_size = args.train_batch_size

        self.valid_batch_size = args.valid_batch_size


        self.pad_token_id = None
    

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("SemevalDataModule")
        parser.add_argument("--promt_style", action="store_true")
        # only for train_dataset
        parser.add_argument("--shuffle_choices", action="store_true")
        parser.add_argument("--train_data_path", type=str, default=None)
        parser.add_argument("--train_batch_size", type=int, default=2)
        parser.add_argument("--valid_batch_size", type=int, default=8)
        parser.add_argument("--num_workers", type=int, default=4)
        return parent_parser

    def setup(self, stage: str):

        train_data = list(np.load(self.train_data_path, allow_pickle=True))
        
        train_size = int(len(train_data) * 0.9)
        valid_size = len(train_data) - train_size

        train_set, valid_set = train_test_split(
            train_data, 
            test_size=0.1,
            random_state=42 
        )
        
        self.trainDataset = SemevalDataset(
            raw_data=train_set,
            encoder_name=self.encoder_name,
            promt_style=self.promt_style,
            include_label=True,
            shuffle_choices=self.shuffle_choices
        )

        self.validDataset = SemevalDataset(
            raw_data=valid_set,
            encoder_name=self.encoder_name,
            promt_style=self.promt_style,
            include_label=True,
            shuffle_choices=False
        )

        self.pad_token_id = self.trainDataset.tokenizer.pad_token_id

    def train_dataloader(self):

        return DataLoader(
            self.trainDataset,
            batch_size=self.train_batch_size,
            collate_fn=collate_fn_factory(self.pad_token_id),
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.validDataset,
            batch_size=self.valid_batch_size,
            collate_fn=collate_fn_factory(self.pad_token_id),
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers
        )

