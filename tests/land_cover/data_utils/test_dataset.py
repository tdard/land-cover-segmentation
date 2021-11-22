import os
from pathlib import Path

from land_cover.data_utils.dataset import LandCoverSegmentationDataset
import albumentations as A


USER_NAME = os.environ["CHALLENGE_USERNAME"]
USER_PWD = os.environ["CHALLENGE_PWD"]
ROOT = Path(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")
)  # This could be a temp file


def test_download_dataset():
    dataset = LandCoverSegmentationDataset(root=ROOT, transform=None, download=True)
    assert 1


def test_instantiate_dataset_no_download():
    dataset = LandCoverSegmentationDataset(root=ROOT, download=False)
    assert 1


def test_dataset_images():
    dataset = LandCoverSegmentationDataset(root=ROOT)
    images = dataset.images

    assert isinstance(images, list)
    assert all(isinstance(i, str) for i in images)


def test_dataset_masks():
    dataset = LandCoverSegmentationDataset(ROOT)
    masks = dataset.masks

    assert isinstance(masks, list)
    assert all(isinstance(m, str) for m in masks)


def test_images_masks_coherence():
    """Check out the length of masks and images data, as well as correct naming"""
    dataset = LandCoverSegmentationDataset(ROOT)

    # Length comparison
    assert len(dataset.images) == len(dataset.masks)

    # Pairwise comparison of names
    assert all(
        i.split(os.sep)[-1] == m.split(os.sep)[-1]
        for (i, m) in zip(dataset.images, dataset.masks)
    )


def test_segmentation_dataset_len():
    dataset = LandCoverSegmentationDataset(ROOT)

    assert len(dataset) > 0  # Requires at least one image and one mask


def test_segmentation_dataset_getitem_no_transform():
    dataset = LandCoverSegmentationDataset(ROOT)
    item = dataset[0]

    assert isinstance(item, tuple) and len(item) == 2


def test_segmentation_dataset_getitem_with_transform():
    transform = A.Compose(
        [
            A.RandomCrop(width=256, height=256),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )

    dataset = LandCoverSegmentationDataset(root=ROOT, transform=transform)

    image, mask = dataset[0]

    assert image.shape == (256, 256, 4)  # R, G, B, N_IR (near infared)
    assert mask.shape == (256, 256, 1)
