from lightning.pytorch.cli import LightningCLI

from src.model import MultipleChoicesModel

from src.data import SemevalDataModule


def cli_main():
    cli = LightningCLI(MultipleChoicesModel, SemevalDataModule)

if __name__ == "__main__":
    cli_main()
