import lightning as L

from argparse import ArgumentParser
from model import MultipleChoicesModel
from pytorch_lightning.loggers import WandbLogger
from data import SemevalDataModule

def main(args):
    L.seed_everything(69, workers=True)

    model = MultipleChoicesModel(args)


    data_module = SemevalDataModule(args)

    loggers = []
    if args.wandb:
        loggers.append(WandbLogger(project="semeval"))

    trainer = L.Trainer.from_argparse_args(
        args,
        loggers=loggers
    )
    
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default=None)
    parser.add_argument("--devices", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser = SemevalDataModule.add_model_specific_args(parser)
    parser = MultipleChoicesModel.add_model_specific_args(parser)
    args = parser.parse_args()

    main(args)