# -*- coding: utf-8 -*- #
"""Data Preprocessing."""
from typing import Tuple, List, Dict

import numpy as np
import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform


class ImageTransformer:
    """
    DataAugmentor for Image Classification.

    Args:
        data_augmentations: List of tuple(method: str, params :dict), each elems pass to albumentations
    """

    def __init__(self, data_augmentations: List[Tuple[str, Dict]]):
        """Initialize."""
        augmentations_list = [
            self._get_augmentation(aug_name)(**params)
            for aug_name, params in data_augmentations]
        self.data_aug = albumentations.Compose(augmentations_list)

    def __call__(self, pair: Tuple[np.ndarray]) -> Tuple[np.ndarray]:
        """Forward"""
        img_arr, label = pair
        return self.data_aug(image=img_arr)["image"], label

    def _get_augmentation(self, aug_name: str) -> ImageOnlyTransform:
        """Get augmentations from albumentations"""
        if hasattr(albumentations, aug_name):
            return getattr(albumentations, aug_name)
        else:
            return eval(aug_name)


class CustomTranspose(ImageOnlyTransform):
    """Given axis, transpose Image Array."""

    def __init__(self, axis: Tuple[int]=(2, 0, 1), always_apply: bool=False, p: float=1.0):
        """Initialize."""
        super().__init__(always_apply, p)
        self.axis = axis

    def apply(self, image: np.ndarray, **params):
        """Apply Transform."""
        return image.transpose(self.axis)


class DuplicateChannel(ImageOnlyTransform):
    """For 1 channel image, duplicate Channel."""

    def __init__(self, n_channel: int=3, always_apply: bool=False, p: float=1.0):
        """Initialize."""
        super().__init__(always_apply, p)
        self.n_channel = n_channel

    def apply(self, image: np.ndarray, **params):
        """
        Apply Transform.

        Note: Input image shape is (Height, Width, Channel).
        """
        return np.repeat(image, self.n_channel, axis=2)


class Inverse(ImageOnlyTransform):
    """For 1 channel image, reverse color."""

    def __init__(self, value_min: int=0, value_max: int=255, always_apply: bool=False, p: float=1.0):
        """Initialize."""
        super().__init__(always_apply, p)
        self.value_min = value_min
        self.value_max = value_max

    def apply(self, image: np.ndarray, **params):
        """
        Apply Transform.

        Note: Input image shape is (Height, Width, Channel).
        """
        return (self.value_max - image) + self.value_min


class RandomErasing(ImageOnlyTransform):
    """Class of RandomErase for Albumentations."""

    def __init__(
        self, s: Tuple[float]=(0.02, 0.4), r: Tuple[float]=(0.3, 2.7),
        mask_value_min: int=0, mask_value_max: int=255,
        always_apply: bool=False, p: float=1.0
    ) -> None:
        """Initialize."""
        super().__init__(always_apply, p)
        self.s = s
        self.r = r
        self.mask_value_min = mask_value_min
        self.mask_value_max = mask_value_max

    def apply(self, image: np.ndarray, **params):
        """
        Apply transform.

        Note: Input image shape is (Height, Width, Channel).
        """
        image_copy = np.copy(image)

        # # decide mask value randomly
        mask_value = np.random.randint(self.mask_value_min, self.mask_value_max + 1)

        h, w, ch = image.shape
        # # decide num of pixcels for mask.
        mask_area_pixel = np.random.randint(h * w * self.s[0], h * w * self.s[1])

        # # decide aspect ratio for mask.
        mask_aspect_ratio = np.random.rand() * self.r[1] + self.r[0]

        # # decide mask hight and width
        mask_height = int(np.sqrt(mask_area_pixel / mask_aspect_ratio))
        if mask_height > h - 1:
            mask_height = h - 1
        mask_width = int(mask_aspect_ratio * mask_height)
        if mask_width > w - 1:
            mask_width = w - 1

        # # decide position of mask.
        top = np.random.randint(0, h - mask_height)
        left = np.random.randint(0, w - mask_width)
        bottom = top + mask_height
        right = left + mask_width
        image_copy[top:bottom, left:right, :].fill(mask_value)

        return image_copy
