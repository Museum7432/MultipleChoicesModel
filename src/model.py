import lightning as L
from torch import nn
import transformers
from torch.nn import functional as F

import torch
from transformers import T5EncoderModel, AutoModel

from allennlp_light.nn.util import (
    batched_index_select,
    masked_softmax,
    masked_log_softmax,
)

import pathlib

from transformers.optimization import get_linear_schedule_with_warmup

import torchmetrics
from argparse import ArgumentParser


class MultipleChoicesModel(L.LightningModule):
    def __init__(
        self,
        encoder_name="google/flan-t5-large",
        lr=1e-5,
        use_last_hidden_state=True,
        loss_threshold=None,
        log_dir=None,
        no_hidden_layer=False,
        lr_scheduler_gamma=0.75
    ):
        """
        :param encoder_name: encoder name; for T5 model, only the encoder will be used.
        :param lr: learning rate.
        :param use_last_hidden_state: use the last hidden state of the encoder's ouput since flanT5 didnt add any special token to the start of the input.
        """
        super().__init__()
        self.save_hyperparameters()

        if "t5" in encoder_name:
            encoder = T5EncoderModel.from_pretrained(encoder_name)

        else:
            encoder = AutoModel.from_pretrained(
                encoder_name,
                add_pooling_layer=False,
            )

        self.encoder = encoder

        self.lr = lr

        self.loss_threshold = loss_threshold

        hidden_size = self.encoder.config.hidden_size

        if no_hidden_layer:
            self.choices_classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, 1),
            )
        else:
            self.choices_classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

        # TODO: num_classes is not always 4
        self.train_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=4
        )
        self.valid_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=4
        )

        self.use_last_hidden_state = use_last_hidden_state

        self.log_dir = log_dir

        self.lr_scheduler_gamma = lr_scheduler_gamma

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = parent_parser.add_argument_group("MultipleChoicesModel")
    #     parser.add_argument("--encoder_name", type=str, default="google/flan-t5-large")
    #     parser.add_argument("--lr", type=float, default=1e-5)
    #     # since flanT5 didnt add any special token to the start of the input
    #     parser.add_argument("--use_last_hidden_state", action="store_true")
    #     return parent_parser

    def forward(self, batch):
        encoded = self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )

        if self.use_last_hidden_state:
            question_state = batched_index_select(
                encoded.last_hidden_state, batch["last_token_indice"].unsqueeze(1)
            )
        else:
            question_state = torch.index_select(
                encoded.last_hidden_state, 1, torch.tensor(0, device=self.device)
            )

        choices_states = batched_index_select(
            encoded.last_hidden_state, batch["indicators_token_offset"]
        )

        # combine question_state with choices_states

        question_choice_rep = torch.cat(
            [question_state.expand_as(choices_states), choices_states], dim=-1
        )

        choices_logits = self.choices_classifier(question_choice_rep).squeeze(2)

        choices_probs = masked_softmax(
            choices_logits.detach(), batch["indicators_token_offset_mask"].detach()
        )

        pred_choices = choices_probs.argmax(dim=1)

        return {
            "choices_logits": choices_logits,
            "choices_probs": choices_probs,
            "pred_choices": pred_choices,
        }

    @staticmethod
    def calc_masked_loss(choices_logits, indicators_token_offset_mask, label):
        logsm = masked_log_softmax(choices_logits, indicators_token_offset_mask)
        logsm = logsm * indicators_token_offset_mask.float()

        loss = F.nll_loss(logsm, label, reduction="sum")

        nof_choices = indicators_token_offset_mask.sum(-1)

        loss = loss / nof_choices

        return loss

    def training_step(self, batch):
        res = self(batch)

        loss = self.calc_masked_loss(
            choices_logits=res["choices_logits"],
            indicators_token_offset_mask=batch["indicators_token_offset_mask"],
            label=batch["label"],
        )

        # TODO: mask all loss values that is smaller than 0.03

        if self.loss_threshold:
            mask = torch.where(loss < self.loss_threshold, 0.0, 1.0)
            loss = loss * mask

        loss = loss.sum()

        self.train_f1(res["pred_choices"], batch["label"])

        self.log("train_f1", self.train_f1)
        self.log("loss", loss.detach(), prog_bar=True)
        return loss

    def validation_step(self, batch):
        res = self(batch)

        loss = self.calc_masked_loss(
            choices_logits=res["choices_logits"],
            indicators_token_offset_mask=batch["indicators_token_offset_mask"],
            label=batch["label"],
        ).sum()

        self.valid_f1(res["pred_choices"].detach(), batch["label"].detach())
        self.log("valid_f1", self.valid_f1)
        self.log("valid_loss", loss.detach())

        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return self.optimizer

        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, mode="min", min_lr=self.lr / 20, verbose=True
        # )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=2,
            gamma=self.lr_scheduler_gamma
        )
        
        scheduler = {
            "scheduler": self.lr_scheduler,
            "interval": "epoch"
        }

        return [self.optimizer], [scheduler]

    def on_test_epoch_start(self):
        self.test_res = []

    @torch.no_grad()
    def test_step(self, batch):
        res = self(batch)

        self.test_res += res["pred_choices"].detach().tolist()

    def on_test_epoch_end(self):
        # self.global_step
        pathlib.Path(self.log_dir, "eval").mkdir(exist_ok=True)

        output_file = pathlib.Path(self.log_dir, "eval", str(self.global_step) + ".txt")

        with open(output_file, "w") as txt_file:
            for i in self.test_res:
                txt_file.write(str(i) + "\n")
