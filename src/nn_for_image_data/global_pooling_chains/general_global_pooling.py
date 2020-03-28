# -*- coding: utf-8 -*- #
"""General Global Pooling Layers."""
from typing import Tuple

import chainer
from chainer import functions


class DummyPooling(chainer.Link):
    """Dummy Layer."""

    def __init__(self, *args) -> None:
        """Initialize."""
        super(DummyPooling, self).__init__()

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward"""
        return x

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        return self(x)


class GlobalPooling(chainer.Link):
    """Global Pooling Layer."""

    def __init__(self, pooling_method: str="avg", axis: Tuple[int]=(2, 3)) -> None:
        """Initialize."""
        super(GlobalPooling, self).__init__()
        self.axis = axis
        func_dict = {
            "avg": functions.average,
            "max": functions.max,
            # "min": functions.min,
        }
        self.pooling_func = func_dict[pooling_method]

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return self.pooling_func(x, axis=self.axis)

    def inference(self, x) -> chainer.Variable:
        """Inference valid or test data."""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return self(x)


class GlobalMaxPooling(GlobalPooling):
    """Global Max Pooling Layer."""

    def __init__(self, axis: Tuple[int]=(2, 3)) -> None:
        """Initialize."""
        super(GlobalMaxPooling, self).__init__(pooling_method="max", axis=axis)


class GlobalAveragePooling(GlobalPooling):
    """Global Average Pooling Layer."""

    def __init__(self, axis: Tuple[int]=(2, 3)) -> None:
        """Initialize."""
        super(GlobalAveragePooling, self).__init__(pooling_method="avg", axis=axis)


class ConcatGlobalAverageMaxPooling(chainer.Link):
    """concatenate multiple concat method."""

    def __init__(self, axis: Tuple[int]=(2, 3)) -> None:
        """Initialize."""
        super(ConcatGlobalAverageMaxPooling, self).__init__()
        self.axis = axis
        self.pooling_func_list = [
            functions.average,
            functions.max]

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, ch, h, w) => return: (bs, 2 * ch)
        return functions.concat(
            [func(x, axis=self.axis) for func in self.pooling_func_list], axis=1)

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        # # x: (bs, ch, h, w) => return: (bs, 2 * ch)
        return self(x)
