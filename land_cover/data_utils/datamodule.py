import os

from pytorch_lightning import LightningDataModule
from typing import Optional, Callable, Union
from os import PathLike

from land_cover.data_utils.dataset import (
    LandCoverSegmentationDataset,
    LandCoverSegmentationDatasetConfig,
)
from torch.utils.data import random_split, DataLoader


class LandCoverSegmentationDataModule(LightningDataModule):
    def __init__(
        self,
        images_dir: PathLike,
        masks_dir: PathLike,
        transform: Optional[Callable] = None,
        train_size: Union[float, int] = 0.9,
        train_batch_size: int = 1,
        val_batch_size: int = 1,
        train_num_workers: int = 0,
        val_num_workers: int = 0,
    ):
        super(LandCoverSegmentationDataModule, self).__init__()

        self.images_dir: PathLike = images_dir
        self.masks_dir: PathLike = masks_dir
        self.transform: Optional[Callable] = transform
        self.train_size: Union[float, int] = train_size
        self.train_batch_size: int = train_batch_size
        self.val_batch_size: int = val_batch_size
        self.train_num_workers: int = train_num_workers
        self.val_num_workers: int = val_num_workers

        self.__sanity_check()

        self.train_dataset: Optional[LandCoverSegmentationDataset] = None
        self.validation_dataset: Optional[LandCoverSegmentationDataset] = None

    def __sanity_check(self):
        if (isinstance(self.train_size, float) and abs(self.train_size) > 1.0) or (
            isinstance(self.train_size, int) and self.train_size < 1
        ):
            raise ValueError(
                "train_size should be a float between 0.0 and 1.0 or an integer greater than 1"
            )

        if not (os.path.exists(self.images_dir) and os.path.exists(self.masks_dir)):
            raise ValueError(
                "One of the paths provided for images and masks does not exist"
            )

    def prepare_data(self) -> None:
        """Nothing to download as the dataset is private for the challenge"""
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        dataset = LandCoverSegmentationDataset(
            self.images_dir, self.masks_dir, self.transform
        )

        if isinstance(
            self.train_size, float
        ):  # In this case, self.train_size is a fraction btw 0.0 and 1.0
            train_size = int(
                LandCoverSegmentationDatasetConfig.TRAINSET_SIZE * self.train_size
            )
        else:
            train_size = self.train_size
        val_size = LandCoverSegmentationDatasetConfig.TRAINSET_SIZE - train_size

        self.train_dataset, self.validation_dataset = random_split(
            dataset, lengths=[train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None

    def teardown(self, stage: Optional[str] = None) -> None:
        pass
