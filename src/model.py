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
from sklearn.metrics import f1_score

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
        loss_threshold_gamma=None,
        log_dir=None,
        no_hidden_layer=False,
        lr_scheduler_gamma=0.75,
    ):
        """
        :param encoder_name: encoder name; for T5 model, only the encoder will be used.
        :param lr: learning rate.
        :param use_last_hidden_state: use the last hidden state of the encoder's ouput since flanT5 didnt add any special token to the start of the input.
        """
        super().__init__()
        self.save_hyperparameters()

        if "t5" in encoder_name.lower() or "ul2" in encoder_name.lower():
            encoder = T5EncoderModel.from_pretrained(encoder_name)

        else:
            encoder = AutoModel.from_pretrained(
                encoder_name,
                add_pooling_layer=False,
            )

        self.encoder = encoder

        self.lr = lr

        self.loss_threshold_gamma = loss_threshold_gamma

        hidden_size = self.encoder.config.hidden_size

        dropout_prob = 0.1

        if no_hidden_layer:
            self.choices_classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size * 2, 1),
            )
        else:
            self.choices_classifier = nn.Sequential(
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, 1),
            )

        # TODO: num_classes is not always 4

        self.valid_f1 = torchmetrics.classification.F1Score(
            task="multiclass", num_classes=4
        )

        self.valid_acc = torchmetrics.Accuracy(task="multiclass", num_classes=4)

        self.use_last_hidden_state = use_last_hidden_state

        self.log_dir = log_dir

        self.lr_scheduler_gamma = lr_scheduler_gamma

        self.effective_trainning_step = 0

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

    def on_train_epoch_start(self):
        self.train_res = {"pred": [], "label": []}

    def on_train_epoch_end(self):
        f1_sc = f1_score(
            self.train_res["pred"], self.train_res["label"], average="micro"
        )

        self.log("train_f1", f1_sc)
        self.train_res = {"pred": [], "label": []}

    def training_step(self, batch):
        if self.loss_threshold_gamma:
            # calculate the loss and then skip those that has lower loss than the threshold
            with torch.no_grad():
                res = self(batch)

                self.train_res["pred"] += res["pred_choices"].detach().tolist()
                self.train_res["label"] += batch["label"].detach().tolist()

                loss = self.calc_masked_loss(
                    choices_logits=res["choices_logits"],
                    indicators_token_offset_mask=batch["indicators_token_offset_mask"],
                    label=batch["label"],
                )

                batch_loss_max = loss.max()

                self.log("loss", loss.detach().sum(), prog_bar=True)

                if not hasattr(self, "loss_threshold"):
                    self.loss_threshold = batch_loss_max
                else:
                    if batch_loss_max > self.loss_threshold * 1.06:
                        self.loss_threshold = batch_loss_max / 1.06
                    elif batch_loss_max > self.loss_threshold * 0.94:
                        self.loss_threshold = (
                            self.loss_threshold * (1 - self.loss_threshold_gamma*3)
                            + batch_loss_max * self.loss_threshold_gamma*3
                        )
                    else:
                        self.loss_threshold = (
                            self.loss_threshold * (1 - self.loss_threshold_gamma)
                            + batch_loss_max * self.loss_threshold_gamma
                        )
                
                self.log("loss_threshold", self.loss_threshold)

                indices = torch.where(loss >= self.loss_threshold)[0]

            # remove those from the batch
            if indices.size(dim=0) == 0:
                # skip this whole batch
                return None

            for k in batch.keys():
                batch[k] = torch.index_select(batch[k], 0, indices)
            


        res = self(batch)

        loss = self.calc_masked_loss(
            choices_logits=res["choices_logits"],
            indicators_token_offset_mask=batch["indicators_token_offset_mask"],
            label=batch["label"],
        )
        loss = loss.sum()

        if self.loss_threshold_gamma is None:
            self.train_res["pred"] += res["pred_choices"].detach().tolist()
            self.train_res["label"] += batch["label"].detach().tolist()
            self.log("loss", loss.detach(), prog_bar=True)
        
        self.effective_trainning_step += batch["label"].size(dim=0)
        return loss

    def validation_step(self, batch):
        res = self(batch)

        loss = self.calc_masked_loss(
            choices_logits=res["choices_logits"],
            indicators_token_offset_mask=batch["indicators_token_offset_mask"],
            label=batch["label"],
        ).sum()

        self.valid_f1(res["pred_choices"].detach(), batch["label"].detach())
        self.valid_acc(res["pred_choices"].detach(), batch["label"].detach())
        self.log("valid_acc", self.valid_acc)
        self.log("valid_f1", self.valid_f1, prog_bar=True)
        self.log("valid_loss", loss.detach())

        self.log("effective_trainning_step", self.effective_trainning_step)
        return loss

    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        return self.optimizer

        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer,
        #     mode="min",
        #     min_lr=self.lr / 20,
        #     verbose=True,
        #     factor=self.lr_scheduler_gamma,
        #     patience=3,
        # )

        # scheduler = {
        #     "scheduler": self.lr_scheduler,
        #     "reduce_on_plateau": True,
        #     "monitor": "valid_loss",
        # }

        # return [self.optimizer], [scheduler]

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer,
        #     step_size=2,
        #     gamma=self.lr_scheduler_gamma
        # )

        # scheduler = {
        #     "scheduler": self.lr_scheduler,
        #     "interval": "epoch"
        # }

        # return [self.optimizer], [scheduler]

    def on_test_epoch_start(self):
        self.test_res = []

    @torch.no_grad()
    def test_step(self, batch):
        res = self(batch)

        self.test_res += res["pred_choices"].detach().tolist()

    def on_test_epoch_end(self):
        # self.global_step
        pathlib.Path(self.log_dir, "eval").mkdir(exist_ok=True)

        output_file = pathlib.Path(self.log_dir, "eval", str(self.effective_trainning_step) + ".txt")

        with open(output_file, "w") as txt_file:
            for i in self.test_res:
                txt_file.write(str(i) + "\n")
