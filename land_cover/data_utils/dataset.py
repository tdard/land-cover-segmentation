import os
from os import PathLike
from typing import Optional, Callable, List, Tuple, Any

import tifffile
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class LandCoverSegmentationDatasetConfig:
    """Class to represent the S2GLC Land Cover Dataset for the challenge,
    with useful metadata and statistics.
    """

    # image size of the images and label masks
    IMG_SIZE = 256
    # the images are RGB+NIR (4 channels)
    N_CHANNELS = 4
    # we have 9 classes + a 'no_data' class for pixels with no labels (absent in the dataset)
    N_CLASSES = 10
    CLASSES = [
        "no_data",
        "clouds",
        "artificial",
        "cultivated",
        "broadleaf",
        "coniferous",
        "herbaceous",
        "natural",
        "snow",
        "water",
    ]
    # classes to ignore because they are not relevant. "no_data" refers to pixels without
    # a proper class, but it is absent in the dataset; "clouds" class is not relevant, it
    # is not a proper land cover type and images and masks do not exactly match in time.
    IGNORED_CLASSES_IDX = [0, 1]

    # The training dataset contains 18491 images and masks
    # The test dataset contains 5043 images and masks
    TRAINSET_SIZE = 18491
    TESTSET_SIZE = 5043

    # for visualization of the masks: classes indices and RGB colors
    CLASSES_COLORPALETTE = {
        0: [0, 0, 0],
        1: [255, 25, 236],
        2: [215, 25, 28],
        3: [211, 154, 92],
        4: [33, 115, 55],
        5: [21, 75, 35],
        6: [118, 209, 93],
        7: [130, 130, 130],
        8: [255, 255, 255],
        9: [43, 61, 255],
    }
    CLASSES_COLORPALETTE = {
        c: np.asarray(color) for (c, color) in CLASSES_COLORPALETTE.items()
    }

    # statistics
    # the pixel class counts in the training set
    TRAIN_CLASS_COUNTS = np.array(
        [
            0,
            20643,
            60971025,
            404760981,
            277012377,
            96473046,
            333407133,
            9775295,
            1071,
            29404605,
        ]
    )
    # the minimum and maximum value of image pixels in the training set
    TRAIN_PIXELS_MIN = 1
    TRAIN_PIXELS_MAX = 24356


class LandCoverSegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: PathLike,
        masks_dir: PathLike,
        transform: Optional[Callable] = None,
    ):
        """

        :param images_dir: The directory of images
        :param masks_dir: The directory of masks
        :param transform: The transform to apply to images and masks

        """
        super(LandCoverSegmentationDataset, self).__init__()
        self.images_dir: PathLike = images_dir
        self.masks_dir: PathLike = masks_dir
        self.transform: Optional[Callable] = transform

        self.__images: List[str] = sorted(
            [
                os.path.join(self.images_dir, image_name)
                for image_name in os.listdir(self.images_dir)
            ]
        )
        self.__masks: List[str] = sorted(
            [
                os.path.join(self.masks_dir, mask_name)
                for mask_name in os.listdir(self.masks_dir)
            ]
        )

    @property
    def images(self) -> List[str]:
        return self.__images

    @property
    def masks(self) -> List[str]:
        return self.__masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        A generic

        :param idx: the index of the (image, tensor) to get

        :return: a tuple of image & mask tensors
        """
        image_path = self.images[idx]
        image: Image = tifffile.imread(image_path)

        mask_path = self.masks[idx]
        mask: Image = np.expand_dims(tifffile.imread(mask_path), axis=-1)

        if self.transform is not None:
            result = self.transform(
                image=image, mask=mask
            )  # Albumentations style transform
            image = result["image"]
            mask = result["mask"]

        return image, mask
