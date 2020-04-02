# -*- coding: utf-8 -*- #
"""Base CNN Models."""
import numpy as np
from typing import Optional, List, Dict

import chainer
from chainercv.links.model.senet import SEResNeXt


class SEResNeXtConv1Mean(SEResNeXt):
    """For non 3-channel or non RGB images, averaging conv1 and copy."""

    def __init__(
        self, n_layer: int, n_class: Optional[int]=None,
        pretrained_model: Optional[str]=None, mean: Optional[List[float]]=None,
        initialW=None, fc_kwargs: Dict={},
        input_channel_num: int=4
    ) -> None:
        """Initialize."""
        super(SEResNeXtConv1Mean, self).__init__(
            n_layer, n_class, pretrained_model, mean, initialW, fc_kwargs=fc_kwargs)

        if pretrained_model is not None:
            w_mean = self.conv1.conv.W.data.mean(axis=1)
            # # # copy the weight to chasnnels, e.g. for (R, G, B, Y).
            # # # (64, 7, 7) * 4 => (64, 4, 7, 7)
            self.conv1.conv.W.data = np.stack([w_mean for _ in range(input_channel_num)], axis=1)

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        h = x
        for name in self.layer_names:
            h = self[name](h)
        return h

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        return self(x)


class SEResNeXt50Conv1MeanTo1ch(SEResNeXtConv1Mean):
    """1channels."""

    def __init__(
        self, n_class: int=None, pretrained_model: Optional[str]=None,
        mean: Optional[List[float]]=None, initialW=None, fc_kwargs: Dict={},
    ) -> None:
        """Initialize."""
        super(SEResNeXt50Conv1MeanTo1ch, self).__init__(
            50, n_class, pretrained_model, mean, initialW,
            fc_kwargs=fc_kwargs, input_channel_num=1)


class SEResNeXt101Conv1MeanTo1ch(SEResNeXtConv1Mean):
    """1channels."""

    def __init__(
        self, n_class: int=None, pretrained_model: Optional[str]=None,
        mean: Optional[List[float]]=None, initialW=None, fc_kwargs: Dict={},
    ) -> None:
        """Initialize."""
        super(SEResNeXt101Conv1MeanTo1ch, self).__init__(
            101, n_class, pretrained_model, mean, initialW,
            fc_kwargs=fc_kwargs, input_channel_num=1)
