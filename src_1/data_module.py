from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
from h11 import Data

import torch
from lightning.pytorch import LightningDataModule
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data import DataLoader, Subset

@dataclass
class KFoldDataModule(LightningDataModule):
    """K-Fold cross validation datamodule.

    Specialized datamodule that can be used for K-Fold cross validation. The first time the `train_dataloader` or
    `test_dataloader` method is call, K folds are generated and the dataloaders are created based on the current fold.

    The input is either a single training dataloader (with an optional validation dataloader) or a lightning datamodule
    that then gets wrapped.

    Args:
        num_folds: Number of folds
        shuffle: Whether to shuffle the data before splitting it into folds
        stratified: Whether to use stratified sampling e.g. for classification we make sure that each fold has the same
            ratio of samples from each class as the original dataset
        train_dataloader: Training dataloader
        val_dataloaders: Validation dataloader(s)
        datamodule: Lightning datamodule

    """

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_folds: int = 5,
        shuffle: bool = False
    ):
        super().__init__()

        if not (isinstance(num_folds, int) and num_folds >= 2):
            raise ValueError("Number of folds must be a positive integer larger than 2")
        self.num_folds = num_folds

        if not isinstance(shuffle, bool):
            raise ValueError("Shuffle must be a boolean value")
        self.shuffle = shuffle

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader

        self.fold_index = 0
        self.splits = None
        self.dataloader_settings = None

    def setup_folds(self) -> None:
        """Implement how folds should be initialized."""
        if self.splits is None:
            labels = None
            splitter = KFold(self.num_folds, shuffle=self.shuffle)
            
            length = len(list(self.train_loader.dataset))
            self.splits = [split for split in splitter.split(list(range(length)), y=labels)]

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader on the current fold."""
        self.setup_folds()
        if self.splits is None:
            raise ValueError("Folds have not been set up")
        train_fold = Subset(self.train_loader.dataset, list(self.splits[self.fold_index][0]))
        return DataLoader(train_fold, **self.dataloader_setting)

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader, which is the same regardless of the fold."""
        return self.val_loader

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader on the current fold."""
        self.setup_folds()
        if self.splits is None:
            raise ValueError("Folds have not been set up")
        test_fold = Subset(self.train_loader.dataset, list(self.splits[self.fold_index][1]))
        return DataLoader(test_fold, **self.dataloader_setting)

    @property
    def dataloader_setting(self) -> dict:
        """Return the settings of the train dataloader."""
        if self.dataloader_settings is None:
            orig_dl = self.train_dataloader()
            self.dataloader_settings = {
                "batch_size": orig_dl.batch_size,
                "num_workers": orig_dl.num_workers,
                "collate_fn": orig_dl.collate_fn,
                "pin_memory": orig_dl.pin_memory,
                "drop_last": orig_dl.drop_last,
                "timeout": orig_dl.timeout,
                "worker_init_fn": orig_dl.worker_init_fn,
                "prefetch_factor": orig_dl.prefetch_factor,
                "persistent_workers": orig_dl.persistent_workers,
            }
        return self.dataloader_settings

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader on the current fold."""
        self.setup_folds()
        test_fold = Subset(self.train_dataset, self.splits[self.fold_index][1])
        return DataLoader(test_fold, **self.dataloader_setting)



    @property
    def dataloader_setting(self) -> dict:
        """Return the settings of the train dataloader."""
        if self.dataloader_settings is None:
            orig_dl = self.train_dataloader()
            self.dataloader_settings = {
                "batch_size": orig_dl.batch_size,
                "num_workers": orig_dl.num_workers,
                "collate_fn": orig_dl.collate_fn,
                "pin_memory": orig_dl.pin_memory,
                "drop_last": orig_dl.drop_last,
                "timeout": orig_dl.timeout,
                "worker_init_fn": orig_dl.worker_init_fn,
                "prefetch_factor": orig_dl.prefetch_factor,
                "persistent_workers": orig_dl.persistent_workers,
            }
        return self.dataloader_settings