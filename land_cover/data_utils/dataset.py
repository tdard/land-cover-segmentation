import os
import pathlib
from os import PathLike
from typing import Optional, Callable, List, Tuple, Any, Union

import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import Dataset

from land_cover.data_utils.download_utils import download_file, login_with_csrf_form
from land_cover.data_utils.file_utils import uncompress_zip


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

    # Files aliases
    X_TRAIN = "X_train.csv"
    Y_TRAIN = "y_train.csv"
    X_TEST = "X_test.csv"
    DATASET = "dataset.zip"

    # URLs
    LOGIN_URL = "https://challengedata.ens.fr/login/"

    DATA_URLS = {
        X_TRAIN: "https://challengedata.ens.fr/participants/challenges/48/download/x-train",  # csv
        Y_TRAIN: "https://challengedata.ens.fr/participants/challenges/48/download/y-train",  # csv
        X_TEST: "https://challengedata.ens.fr/participants/challenges/48/download/x-test",  # csv
        DATASET: "https://challengedata.ens.fr/participants/challenges/48/download/supplementary-files",  # Zip
    }

    # Dataset structure
    TRAIN_IMAGES_FOLDER = os.path.join("train", "images")
    TRAIN_MASKS_FOLDER = os.path.join("train", "masks")

    def __init__(
        self,
        root: Union[str, PathLike],
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """
        :param root: The directory in which to put the dataset
        :param transform: The transform to apply to images and masks
        :param download: If you want to download the dataset. You need to have set the environment variables
        CHALLENGE_USERNAME and CHALLENGE_PWD with your credentials on the website challengedata.ens.fr

        """
        super(LandCoverSegmentationDataset, self).__init__()

        self.root: Union[str, PathLike] = root
        self.transform: Optional[Callable] = transform

        self.images_dir: str = os.path.join(self.dataset_path, self.TRAIN_IMAGES_FOLDER)
        self.masks_dir: str = os.path.join(self.dataset_path, self.TRAIN_MASKS_FOLDER)

        if (not download) and not (
            os.path.isdir(self.images_dir) or os.path.isdir(self.masks_dir)
        ):
            raise ValueError("The directories for the dataset do not exist")

        if download:
            self.download()

        self.uncompress_dataset()

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
    def dataset_path(self) -> str:
        return os.path.join(self.root, pathlib.Path(self.DATASET).stem)

    def download(self):
        if "CHALLENGE_PWD" not in os.environ or "CHALLENGE_USERNAME" not in os.environ:
            raise KeyError(
                "User should provide his credential through 'CHALLENGE_PWD' and 'CHALLENGE_USERNAME' "
                "variables"
            )

        session = login_with_csrf_form(
            self.LOGIN_URL,
            username=os.environ["CHALLENGE_USERNAME"],
            password=os.environ["CHALLENGE_PWD"],
            username_form_name="username",
            password_form_name="password",
        )

        for filename, url in self.DATA_URLS.items():
            download_file(url, session, self.root, filename)

    def uncompress_dataset(self):
        dataset_zip_path = os.path.join(self.root, self.DATASET)
        if not os.path.isdir(self.dataset_path) and os.path.exists(dataset_zip_path):
            uncompress_zip(dataset_zip_path, self.root, keep_zip_name=False)

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
