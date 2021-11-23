import os
from pathlib import Path

import pytest

from land_cover.data_utils.datamodule import LandCoverSegmentationDataModule
from land_cover.data_utils.dataset import LandCoverSegmentationDatasetConfig
from land_cover.data_utils.augmentations import to_float32


import albumentations.pytorch.transforms as T
import albumentations as A

import torch


USER_NAME = os.environ["CHALLENGE_USERNAME"]
USER_PWD = os.environ["CHALLENGE_PWD"]
ROOT = Path(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)  # This could be a temp file


def test_init_and_setup_datamodule():
    dm = LandCoverSegmentationDataModule(root=ROOT, download=True)
    dm.setup()
    assert 1


def test_train():
    dm = LandCoverSegmentationDataModule(root=ROOT, train_size=0.9)
    dm.setup()

    train_dataset = dm.train_dataset
    val_dataset = dm.validation_dataset

    assert len(train_dataset) == int(
        0.9 * LandCoverSegmentationDatasetConfig.TRAINSET_SIZE
    )
    assert len(val_dataset) == LandCoverSegmentationDatasetConfig.TRAINSET_SIZE - len(
        train_dataset
    )


@pytest.mark.parametrize("num_iterations,train_batch_size", [(1, 1), (4, 1)])
def test_iterations(num_iterations: int, train_batch_size: int):
    dm = LandCoverSegmentationDataModule(
        root=ROOT,
        train_size=0.5,
        transform=A.Compose(
            [
                A.Lambda(name="to_float32", image=to_float32),
                T.ToTensorV2(transpose_mask=True),
            ]
        ),
        train_batch_size=train_batch_size,
    )
    dm.setup()

    for i, x in enumerate(dm.train_dataloader()):
        image, mask = x

        assert isinstance(image, torch.Tensor)
        assert image.ndim == 4
        assert mask.ndim == 4
        assert len(image) == train_batch_size
        assert len(mask) == train_batch_size

        if i == num_iterations:
            break
