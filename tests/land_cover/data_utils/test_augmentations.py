import numpy as np
import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from land_cover.data_utils.augmentations import (
    select_rgb_channels_only,
    cast,
    to_float32,
    min_max_normalization,
)
import pytest
from functools import partial


def test_select_rgb_channels_only():
    image = np.random.randn(224, 224, 4)
    mask = np.random.randn(224, 224, 1)

    transform = A.Compose(
        [A.Lambda(name="select_rgb_channels_only", image=select_rgb_channels_only)]
    )

    result = transform(image=image, mask=mask)

    transformed_image = result["image"]
    transformed_mask = result["mask"]

    assert transformed_image.shape == (224, 224, 3)
    assert transformed_mask.shape == (224, 224, 1)


@pytest.mark.parametrize("dtype", [np.float32])
def test_cast(dtype):
    image = np.random.randint(11, 999, dtype=np.uint16)
    result = cast(image, dtype)
    assert result.dtype == dtype


def test_to_float32():
    image = np.random.randint(11, 999, dtype=np.uint16)
    result = to_float32(image)
    assert result.dtype == np.float32


def test_masks_keep_consistent_dtype_when_input_is_normalized():
    augmentations = A.Compose(
        [
            # Cast image to float32 before normalization
            A.Lambda(name="to_float32", image=to_float32),
            # Normalize image with min-max normalization
            A.Lambda(
                name="min_max_normalization",
                image=partial(min_max_normalization, max_=24356, min_=1),
            ),
            # Set image to Tensor
            ToTensorV2(transpose_mask=True),
        ]
    )

    image = np.random.randint(low=1, high=24356, size=(224, 224, 4), dtype=np.uint16)
    mask = np.random.randint(low=10, high=11, size=(224, 224, 1), dtype=np.uint8)

    result = augmentations(image=image, mask=mask)

    res_image, res_mask = result["image"], result["mask"]

    assert res_mask.dtype == torch.uint8
    assert res_mask.shape == (1, 224, 224)

    assert res_image.dtype == torch.float32
    assert res_image.shape == (4, 224, 224)
    assert torch.max(res_image) <= 1.0
    assert torch.min(res_image) >= 0
