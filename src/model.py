import lightning as L
from torch import nn
import transformers
from torch.nn import functional as F

import torch
from transformers import T5EncoderModel

from allennlp_light.nn.util import batched_index_select, masked_softmax, masked_log_softmax

from transformers.optimization import get_linear_schedule_with_warmup

import torchmetrics
from argparse import ArgumentParser


class MultipleChoicesModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()


        self.encoder = T5EncoderModel.from_pretrained(args.encoder_name)

        if args.gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        self.lr = args.lr

        hidden_size = self.encoder.config.hidden_size

        self.choices_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1)
        )

        # TODO: num_classes is not always 4
        self.train_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=4)

        self.use_last_hidden_state = args.use_last_hidden_state
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MultipleChoicesModel")
        parser.add_argument("--encoder_name", type=str, default="google/flan-t5-large")
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--gradient_checkpointing", action="store_true")
        # since flanT5 didnt add any special token to the start of the input
        parser.add_argument("--use_last_hidden_state", action="store_true")
        return parent_parser

    def forward(self, batch):
        encoded = self.encoder(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        
        if self.use_last_hidden_state:
            question_state = torch.index_select(encoded.last_hidden_state, 1, batch["last_token_indice"]).squeeze(1)
        else:
            question_state = torch.index_select(encoded.last_hidden_state, 1, torch.tensor(0)).squeeze(1)
        
        choices_states = batched_index_select(
            encoded.last_hidden_state, batch["indicators_token_offset"]
        )

        # combine question_state with choices_states

        tmp = question_state.unsqueeze(1).expand_as(choices_states)

        question_choice_rep = torch.cat([tmp, choices_states], dim = -1)

        choices_logits = self.choices_classifier(question_choice_rep).squeeze(2)

        choices_probs = masked_softmax(choices_logits.detach(), batch["indicators_token_offset_mask"].detach())

        pred_choices = choices_probs.argmax(dim=1)

        return {
            "choices_logits": choices_logits,
            "choices_probs": choices_probs,
            "pred_choices": pred_choices
        }

    def training_step(self, batch):
        res = self(batch)

        logsm = masked_log_softmax(res["choices_logits"], batch["indicators_token_offset_mask"])

        logsm = logsm * batch["indicators_token_offset_mask"].float()

        loss = F.nll_loss(
            logsm,
            batch["label"],
            reduction="sum"
        )

        nof_choices = batch["indicators_token_offset_mask"].sum(-1)

        loss = (loss / nof_choices).sum()

        self.train_f1(res["pred_choices"], batch["label"])

        self.log('train_f1', self.train_f1)
        self.log("loss", loss.detach())

        print(loss.detach())

        return loss

    def configure_optimizers(self):

        self.optimizer = AdamW(self.parameters(), lr=self.lr)

        return self.optimizer
