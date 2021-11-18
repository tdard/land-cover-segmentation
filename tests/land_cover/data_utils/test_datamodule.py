import os
from pathlib import Path

import pytest

from land_cover.data_utils.datamodule import LandCoverSegmentationDataModule
from land_cover.data_utils.dataset import LandCoverSegmentationDatasetConfig
from land_cover.data_utils.augmentations import to_float32


import albumentations.pytorch.transforms as T
import albumentations as A

import torch

CURRENT_FOLDER: Path = Path(os.path.dirname(__file__))
DATA_FOLDER = CURRENT_FOLDER / ".." / ".." / ".." / "data"

IMAGES_PATH = DATA_FOLDER / "dataset_UNZIPPED" / "train" / "images"
MASKS_PATH = DATA_FOLDER / "dataset_UNZIPPED" / "train" / "masks"


def test_init_and_setup_datamodule():
    dm = LandCoverSegmentationDataModule(images_dir=IMAGES_PATH, masks_dir=MASKS_PATH)
    dm.setup()
    assert 1


def test_train():
    dm = LandCoverSegmentationDataModule(
        images_dir=IMAGES_PATH, masks_dir=MASKS_PATH, train_size=0.9
    )
    dm.setup()

    train_dataset = dm.train_dataset
    val_dataset = dm.validation_dataset

    assert len(train_dataset) == int(
        0.9 * LandCoverSegmentationDatasetConfig.TRAINSET_SIZE
    )
    assert len(val_dataset) == LandCoverSegmentationDatasetConfig.TRAINSET_SIZE - len(
        train_dataset
    )


@pytest.mark.parametrize("num_iterations,train_batch_size", [(1, 4), (4, 10)])
def test_iterations(num_iterations: int, train_batch_size: int):
    dm = LandCoverSegmentationDataModule(
        images_dir=IMAGES_PATH,
        masks_dir=MASKS_PATH,
        train_size=0.9,
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
