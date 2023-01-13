import pytorch_lightning as pl
from torch import nn, optim
from torch.functional import F
import torch


class AutoEncoder(pl.LightningModule):
    def __init__(self, encoded_space_dim, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.example_input_array = torch.Tensor(1, 1, 28, 28)
        self.encoder = Encoder(encoded_space_dim=encoded_space_dim)
        self.decoder = Decoder(encoded_space_dim=encoded_space_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, _ = batch
        encoded = self.encoder(x)
        x_hat = self.decoder(encoded)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)

        # log before and after images
        if batch_idx % 100 == 0:
            self.logger.experiment.add_image("input", x[0, 0, :, :], self.current_epoch, dataformats="HW")
            self.logger.experiment.add_image("output", x_hat[0, 0, :, :], self.current_epoch, dataformats="HW")

        return loss


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2),
            nn.ReLU(),
            ResidualBlock(),
            ResidualBlock(),
            nn.Conv2d(in_channels=256, out_channels=encoded_space_dim, kernel_size=1, stride=1),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=encoded_space_dim, out_channels=256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(),
            ResidualBlock(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=1, kernel_size=4, stride=2),
            nn.Sigmoid(),  # Just a guess?
        )

    def forward(self, x):
        return self.decoder(x)


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x_in):
        x = self.relu(x_in)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + x_in
