import os
import random
from matplotlib import pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision.utils import save_image
#from tsnecuda import TSNE
import matplotlib as mpl
from sklearn.manifold import TSNE

from st_out_supress import suppress_stdout_stderr


class VAE(LightningModule):
    def __init__(self, beta=1.0):
        super().__init__()

        self.beta = beta
        self.dataset = MNIST
        self.out_channel = 1
        self.encoder = torch.nn.Sequential(
            nn.Conv2d(self.out_channel, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.hidden_mu = nn.Linear(512, 128)
        self.hidden_logvar = nn.Linear(512, 128)

        self.decoder_in = nn.Linear(128, 512)
        self.decoder = torch.nn.Sequential(
            nn.ConvTranspose2d(
                512, 256, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                256, 128, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                128, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(
                32,
                self.out_channel,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.Tanh(),
        )
        self.data_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
            ]
        )
        self.tsne = TSNE(
            n_components=2,
            perplexity=15,
            learning_rate=10,
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std).type_as(mu)
        return mu + eps * std

    def encode(self, x):
        hidden = self.encoder(x)
        hidden = torch.flatten(hidden, 1)
        mu = self.hidden_mu(hidden)
        logvar = self.hidden_logvar(hidden)
        return mu, logvar

    def decode(self, z):
        out = self.decoder_in(z)
        out = out.view(-1, 512, 1, 1)
        return self.decoder(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        batch_size = x.size(0)
        # x = x.view(x.size(0), -1)

        mu, logvar = self.encode(x)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        z = self.reparameterize(mu, logvar)
        x_reconstruct = self.decode(z)
        # reconstruct_loss_type = nn.MSELoss()
        reconstruct_loss = self.get_reconstruct_loss(x_reconstruct, x)

        loss = reconstruct_loss + kl_loss * self.beta
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_kl_loss",
            kl_loss / batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_rc_loss",
            reconstruct_loss / batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = x.size(0)
        # x = x.view(x.size(0), -1)

        mu, logvar = self.encode(x)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum()
        z = self.reparameterize(mu, logvar)
        x_reconstruct = self.decode(z)
        # reconstruct_loss_type = nn.MSELoss()
        reconstruct_loss = self.get_reconstruct_loss(x_reconstruct, x)

        loss = reconstruct_loss + kl_loss * self.beta
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "val_kl_loss",
            kl_loss / batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val_rc_loss",
            reconstruct_loss / batch_size,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return x_reconstruct, y, loss, mu, logvar

    def get_reconstruct_loss(self, x_reconstruct, x):
        # print(x.shape, x.mean(), x.std())
        # print(x_reconstruct.shape, x_reconstruct.mean(), x_reconstruct.std())
        # bce = nn.functional.binary_cross_entropy(x_reconstruct, x, reduction="sum")
        mse = nn.functional.mse_loss(x_reconstruct, x, reduction="sum")
        log_sigma_opt = 0.5 * mse.log()
        r_loss = (
            0.5 * torch.pow((x - x_reconstruct) / log_sigma_opt.exp(), 2)
            + log_sigma_opt
        ).sum()
        return r_loss

    def train_dataloader(self):
        train_data = self.dataset(
            "data/", download=True, train=True, transform=self.data_transform
        )
        return DataLoader(train_data, batch_size=16)

    def val_dataloader(self):
        val_data = self.dataset(
            "data/", download=True, train=False, transform=self.data_transform
        )
        return DataLoader(val_data, batch_size=16)

    @staticmethod
    def scale_image(img):
        out = (img + 1) / 2
        return out

    def validation_epoch_end(self, outputs):
        if not os.path.exists("vae_images"):
            os.makedirs("vae_images")
        choice = random.choice(outputs)  # Choose a random batch from outputs
        output_sample, y, loss, output_mu, output_logvar = choice
        output_latent = self.reparameterize(output_mu, output_logvar)
        output_sample = output_sample.reshape(-1, self.out_channel, 32, 32)
        output_sample = self.scale_image(output_sample)
        save_image(output_sample, f"vae_images/epoch_{self.current_epoch+1}.png")
        latent_image = self.get_latent_image(output_latent.cpu(), y.cpu())
        latent_image.savefig(f"vae_images/epoch_{self.current_epoch+1}_space.png")
        plt.close()

    def get_latent_image(self, output_latent, y=None):
        #with suppress_stdout_stderr() as _:
            # with contextlib.redirect_stderr(os.devnull) as _:
        #    latent_tsne = self.tsne.fit_transform(output_latent.cpu())
        latent_tsne = self.tsne.fit_transform(output_latent.cpu())

        fig, ax = plt.subplots()
        cmap = plt.cm.tab20b
        cmaplist = [cmap(i) for i in range(cmap.N)]
        bounds = np.linspace(0, 10, 11)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        scat = ax.scatter(*latent_tsne.T, c=y, cmap=cmap, norm=norm)
        cb = plt.colorbar(scat, ticks=list(range(10)))
        cb.set_label("Labels")
        ax.set_title("TSNE plot for VAE Latent Space colour coded by Labels")
        return fig


if __name__ == "__main__":
    trainer = Trainer(accelerator="mps", devices=1, auto_lr_find=False)
    trainer.fit(VAE(beta=1.2))
