import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule

# Пример простой модели с добавлением гауссовского шума и constant scheduler
# Допустим, у нас есть простой автоэнкодер или просто сетка из линейных слоёв,
# на вход которой подаются данные, к которым добавляется шум.
# На выходе модель должна приблизиться к исходным данным.

class SimpleGaussianDiffusionModel(LightningModule):
    def __init__(self, input_dim=100, hidden_dim=64, lr=1e-4, weight_decay=1e-5, noise_std=0.1):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )
        self.noise_std = noise_std
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        # Проход через простую сеть
        h = self.encoder(x)
        return self.decoder(h)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        # Добавляем гауссовский шум к входным данным
        noise = torch.randn_like(x) * self.noise_std
        noisy_x = x + noise
        pred = self(noisy_x)
        loss = ((pred - x) ** 2).mean()
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Constant scheduler: без изменения learning rate
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
