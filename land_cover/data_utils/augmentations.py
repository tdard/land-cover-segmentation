import numpy as np


def select_rgb_channels_only(image: np.ndarray, **kwargs) -> np.ndarray:
    """
    Select RGB array of a channel-last, 4D image.

    :param image:
    :param kwargs:
    :return:
    """
    return image[..., :-1]


def cast(image: np.ndarray, dtype, **kwargs) -> np.ndarray:
    """
    Cast to float32

    :param image:
    :param kwargs:
    :return:
    """
    return image.astype(dtype)


def to_float32(image: np.ndarray, **kwargs) -> np.ndarray:
    return cast(image, np.float32)


def min_max_normalization(
    image: np.ndarray, max_: int, min_: int = 1, **kwargs
) -> np.ndarray:
    return (image - min_) / (max_ - min_)
