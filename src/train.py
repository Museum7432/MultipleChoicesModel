import lightning as L

from argparse import ArgumentParser
from model import MultipleChoicesModel
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from data import SemevalDataModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch


def main(args):
    L.seed_everything(69, workers=True)


    torch.set_float32_matmul_precision(args.float32_matmul_precision)


    data_module = SemevalDataModule(
        train_data_path = args.train_data_path,
        eval_data_path = args.eval_data_path,
        encoder_name = args.encoder_name,
        promt_style = args.promt_style,
        shuffle_choices = args.shuffle_choices,
        num_workers = args.num_workers,
        train_batch_size = args.train_batch_size,
        valid_batch_size= args.valid_batch_size,
        debug= args.debug,
        reduce_choices=args.reduce_choices
    )
    

    loggers = []
    if args.wandb:
        loggers.append(WandbLogger(project="semeval"))

    tb = TensorBoardLogger("./")
    loggers.append(tb)
    
    log_dir = tb.log_dir

    lr_monitor = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        monitor='valid_acc',
        mode="max",
        save_top_k=2,
        save_last=True,
        save_weights_only=True,
        dirpath=log_dir+"/checkpoint"
    )
    

    callbacks = [lr_monitor, checkpoint_callback]

    trainer = L.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        logger=loggers,
        callbacks=callbacks,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        max_epochs=args.max_epochs,
        log_every_n_steps=5
    )

    model = MultipleChoicesModel(
        encoder_name=args.encoder_name,
        lr=args.lr,
        use_last_hidden_state=args.use_last_hidden_state,
        log_dir=log_dir,
        no_hidden_layer=args.no_hidden_layer,
        loss_threshold_gamma=args.loss_threshold_gamma,
        lr_scheduler_gamma=args.lr_scheduler_gamma
    )
    
    trainer.fit(model, datamodule=data_module)

    trainer.test(ckpt_path='last', datamodule=data_module)

    trainer.test(ckpt_path="best", datamodule=data_module)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default=1)
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--float32_matmul_precision", type=str, default="highest")

    parser.add_argument("--accumulate_grad_batches",type=int, default=1)
    parser.add_argument("--max_epochs",type=int, default=-1)


    
    parser.add_argument("--encoder_name", type=str, default="google/flan-t5-large")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler_gamma", type=float, default=0.75)
    parser.add_argument("--loss_threshold_gamma", type=float, default=None)
    # since flanT5 didnt add any special token to the start of the input
    parser.add_argument("--use_last_hidden_state", action="store_true")
    parser.add_argument("--no_hidden_layer", action="store_true")

    parser.add_argument("--promt_style", action="store_true")
    # only for train_dataset
    parser.add_argument("--shuffle_choices", action="store_true")
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--eval_data_path", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--valid_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--reduce_choices", action="store_true")
    
    args = parser.parse_args()

    main(args)