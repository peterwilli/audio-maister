
import lightning.pytorch as pl
from torch.utils.data import DataLoader
import torch

class PreProcessedDataModule(pl.LightningDataModule):
    def __init__(self, dataset_train, dataset_val, n_workers: int = 4, batch_size: int = 32):
        super(PreProcessedDataModule, self).__init__()
        self.n_workers = n_workers
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.batch_size = batch_size

    def transform(self, example):
        return { k: [torch.tensor(example[k][i]["array"]).unsqueeze(1) for i in range(len(example[k]))] for k in example }

    def train_dataloader(self) -> DataLoader:
        dataset = self.dataset_train.with_transform(self.transform)
        return DataLoader(dataset, num_workers=self.n_workers, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        dataset = self.dataset_val.with_transform(self.transform)
        return DataLoader(dataset, num_workers=self.n_workers, batch_size=4, shuffle=False)