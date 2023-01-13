import click
import yaml

import torch

from torchvision import transforms
from torchvision.datasets import MNIST

from mnist_autoencoder import AutoEncoder

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary


@click.group()
def cli():
    pass


@cli.command()
@click.option("--config", default="config.yaml", help="Path to config file")
def train_mnist(config):
    """Train the MNIST autoencoder"""

    # Load the config file
    with open(config, "r") as f:
        config = yaml.safe_load(f)

    print(config)

    # Load the dataset
    train_dataset = MNIST(
        root=config["dataset"]["path"],
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    # Create the dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )

    # Create the model
    model = AutoEncoder(encoded_space_dim=config["AE"]["D"])

    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        accelerator="gpu",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="train_loss", patience=3, verbose=True, mode="min"),
            ModelSummary(max_depth=-1),
        ],
    )

    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    cli()
