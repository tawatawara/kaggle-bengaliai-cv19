# -*- coding: utf-8 -*- #
"""Wrapper for Pooling Layers."""
from typing import Tuple, Dict
import chainer

from .general_global_pooling import DummyPooling, GlobalMaxPooling, GlobalAveragePooling
from .general_global_pooling import ConcatGlobalAverageMaxPooling
from .global_squeeze_and_excitation_pooling import GlobalcSEPooling, GlobalsSEPooling, GlobalscSEPooling


POOLING_DICT = {
    "Duumy": DummyPooling,
    "MaxPool": GlobalMaxPooling,
    "AvgPool": GlobalAveragePooling,
    "AvgMaxPool": ConcatGlobalAverageMaxPooling,
    "cSEAvgPool": GlobalcSEPooling,
    "sSEAvgPool": GlobalsSEPooling,
    "scSEAvgPool": GlobalscSEPooling}


class PoolingLayer(chainer.Chain):
    """Wrapper for Pooling Layer."""

    def __init__(self, pooling_layer: str="AvgPool", pooling_kwargs: Dict=dict()) -> None:
        """Initialize."""
        super(PoolingLayer, self).__init__()

        with self.init_scope():
            self.pool = POOLING_DICT[pooling_layer](**pooling_kwargs)

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward"""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return self.pool(x)

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return self(x)


class MultiHeadPoolingLayer(chainer.Chain):
    """Wrapper for Pooling Layer, which output multiple Variables"""

    unlearnable_poolings = set(
        ["Duumy", "MaxPool", "AvgPool", "AvgMaxPool"])

    def __init__(self, pooling_layer: str="AvgPool", pooling_kwargs: Dict=dict(), output_num: int=3) -> None:
        """Initialize."""
        super(MultiHeadPoolingLayer, self).__init__()
        self.pooling_method = pooling_layer
        self.output_num = output_num

        with self.init_scope():
            if self.pooling_method in self.unlearnable_poolings:
                self.pool = POOLING_DICT[pooling_layer]
            else:
                for i in range(self.output_num):
                    setattr(
                        self, "pool{}".format(i + 1), POOLING_DICT[pooling_layer](**pooling_kwargs))

    def __call__(self, x: chainer.Variable) -> Tuple[chainer.Variable]:
        """Forward"""
        # # x: (bs, ch, h, w) => h: (bs, ch) => ret_list: [(bs, ch),] * output_num
        if self.pooling_method in self.unlearnable_poolings:
            h = self.pool(x)
            ret_list = [h for i in range(self.output_num)]
        else:
            ret_list = [
                getattr(self, "pool{}".format(i + 1))(x) for i in range(self.output_num)]

        return tuple(ret_list)

    def inference(self, x: chainer.Variable) -> Tuple[chainer.Variable]:
        """Inference valid or test data."""
        # # x: (bs, ch, h, w) => return: ((bs, ch),) * output_num
        return self(x)


class TripleHeadPoolingLayer(MultiHeadPoolingLayer):
    """Wrapper for Pooling Layer, which output multiple Variables"""

    unlearnable_poolings = set(
        ["Duumy", "MaxPool", "AvgPool", "AvgMaxPool"])

    def __init__(self, pooling_layer: str="AvgPool", pooling_kwargs: Dict=dict()) -> None:
        """Initialize."""
        super(TripleHeadPoolingLayer, self).__init__(
            pooling_layer=pooling_layer, pooling_kwargs=pooling_kwargs, output_num=3)
