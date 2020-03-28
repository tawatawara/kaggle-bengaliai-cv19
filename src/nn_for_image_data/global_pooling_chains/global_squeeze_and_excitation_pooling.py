# -*- coding: utf-8 -*- #
"""Global Squeeze & Excitation Pooling Layers."""

from typing import Tuple

import chainer
from chainercv.links import SEBlock
from chainer import functions, links


class SSEBlock(chainer.Chain):
    """channel `S`queeze and `s`patial `E`xcitation Block."""

    def __init__(self):
        """Initialize."""
        super(SSEBlock, self).__init__()

        with self.init_scope():
            self.channel_squeeze = links.Convolution2D(
                in_channels=None, out_channels=1, ksize=1, stride=1)

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, ch, h, w) => h: (bs, 1, h, w)
        h = functions.sigmoid(self.channel_squeeze(x))
        # # x, h => return: (bs, ch, h, w)
        return x * h


class GlobalsSEPooling(chainer.Chain):
    """Global Average Pooling with sSE Module."""

    def __init__(self, axis: Tuple[int]=(2, 3)):
        """Initialize."""
        super(GlobalsSEPooling, self).__init__()
        self.axis = axis
        with self.init_scope():
            self.sse = SSEBlock()

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return functions.average(self.sse(x), axis=self.axis)

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return self(x)


class GlobalcSEPooling(chainer.Chain):
    """Global Average Pooling with cSE Module(original SE Module)."""

    def __init__(
        self, n_channel: int=2048, ratio: int=2, axis: Tuple[int]=(2, 3)
    ) -> None:
        """Initialize."""
        super(GlobalcSEPooling, self).__init__()
        self.axis = axis
        with self.init_scope():
            self.cse = SEBlock(n_channel, ratio)

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return functions.average(self.cse(x), axis=self.axis)

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return self(x)


class GlobalscSEPooling(chainer.Chain):
    """Global Average Pooling with scSE Module."""

    def __init__(
        self, n_channel: int=2048, ratio: int=2, axis: Tuple[int]=(2, 3)
    ) -> None:
        """Initialize."""
        super(GlobalscSEPooling, self).__init__()
        self.axis = axis
        with self.init_scope():
            self.sse = SSEBlock()
            self.cse = SEBlock(n_channel, ratio)

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return functions.average(self.cse(x) + self.sse(x), axis=self.axis)

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        # # x: (bs, ch, h, w) => return: (bs, ch)
        return self(x)
