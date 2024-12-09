import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from model import SimpleGaussianDiffusionModel
import os

# Предположим, что data/weights.npy и data/good_paths.npy имеют одинаковую длину
# и что данные представлены в формате [кол-во_примеров, размер_входа].
# Например, если вход - это вектор длины input_dim, то веса и пути должны быть [N, input_dim].
# Если это матрицы [N, M, M], то вы можете их расплющить до [N, M*M] или модифицировать модель.

class GraphDataset(Dataset):
    def __init__(self, weights_path, good_paths_path):
        self.weights = np.load(weights_path)        # shape: [1000, 30, 30]
        self.good_paths = np.load(good_paths_path)  # shape: [1000, 30]
        
        self.weights = torch.from_numpy(self.weights).float()
        self.good_paths = torch.from_numpy(self.good_paths).float()

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, idx):
        return self.weights[idx], self.good_paths[idx]




def main():
    

    

    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Загрузка данных
    train_dataset = GraphDataset(os.path.join(base_dir, "data/weights.npy"), os.path.join(base_dir, "data/good_paths.npy"))

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    for batch in train_loader:
        weights_batch, good_paths_batch = batch
        print(f"Weights batch shape: {weights_batch.shape}")
        print(f"Good paths batch shape: {good_paths_batch.shape}")
        break

        
    # Инициализация модели
    # Предположим, что размер входа соответствует форме данных.
    # Если weights.npy имеет форму [N, D], то input_dim = D
    input_dim = train_dataset.weights.shape[1]
    model = SimpleGaussianDiffusionModel(input_dim=input_dim, hidden_dim=128, lr=1e-4, weight_decay=1e-5, noise_std=0.1)

    # Перепишем training_step модели так, чтобы использовать good_paths как цель:
    # Для этого переопределим метод в наследнике:
    class ModelWithTargets(SimpleGaussianDiffusionModel):
        def training_step(self, batch, batch_idx):
            weights, good_paths = batch
            noise = torch.randn_like(weights) * self.noise_std
            noisy_input = weights + noise
            pred = self(noisy_input)
            print(f"pred shape: {pred.shape}")

            exit()
            loss = ((pred - good_paths) ** 2).mean()
            self.log("train_loss", loss, prog_bar=True)
            return loss

    model = ModelWithTargets(input_dim=input_dim, hidden_dim=128, lr=1e-4, weight_decay=1e-5, noise_std=0.1)

    # Настройка тренера
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )

    # Обучение
    trainer.fit(model, train_loader)


if __name__ == "__main__":
    main()
