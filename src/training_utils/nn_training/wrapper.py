# -*- coding: utf-8 -*- #
"""wrapper for training and prediction."""

# from itertools import product
from typing import Union, Tuple, Dict, Optional

# import numpy as np

import chainer
from chainer import links
from chainer import reporter
from chainercv.links import PickableSequentialChain


class ImageClassificationModel(PickableSequentialChain):
    """Predictor Wrapper."""

    def __init__(
        self,
        extractor: chainer.Chain,
        global_pooling: Optional[chainer.Chain],
        classifier: chainer.Chain
    ) -> None:
        """Initialize."""
        super(ImageClassificationModel, self).__init__()

        with self.init_scope():
            self.extractor = extractor
            if global_pooling is not None:
                self.global_pooling = global_pooling
            self.classifier = classifier

    def __call__(
        self, x: chainer.Variable
    ) -> Union[chainer.Variable, Tuple[chainer.Variable]]:
        """Farward."""
        h = x
        for layer_name in self.layer_names:
            h = self[layer_name](h)
        return h

    def inference(
        self, x: chainer.Variable
    ) -> Union[chainer.Variable, Tuple[chainer.Variable]]:
        """Inference Test Data."""
        return self(x)


class CustomClassifier(links.Classifier):
    """Custom Training Wrapper for classification model."""

    def __init__(
        self,
        predictor: PickableSequentialChain,
        lossfun: object,
        evalfun_dict: Dict[str, object],
    ) -> None:
        """Initialize"""
        super(CustomClassifier, self).__init__(predictor, lossfun)
        self.compute_accuracy = False
        for name in evalfun_dict.keys():
            setattr(self, name, None)
        self.evalfun_dict = evalfun_dict

    def __call__(self, *in_arrs: Tuple[chainer.Variable]) -> float:
        """Forward."""
        # # Foward: calc loss.
        loss = super().__call__(*in_arrs)
        # print(type(loss))
        return loss

    def evaluate(self, *in_arrs: Tuple[chainer.Variable]) -> None:
        """Calc loss and evaluation metric."""
        for name in self.evalfun_dict.keys():
            setattr(self, name, None)
        loss = self(*in_arrs)
        for name, evalfun in self.evalfun_dict.items():
            setattr(self, name, evalfun(self.y, in_arrs[-1]))
            reporter.report({name: getattr(self, name)}, self)
        del loss
