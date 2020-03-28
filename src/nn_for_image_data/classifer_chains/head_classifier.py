# -*- coding: utf-8 -*- #
"""Wrapper of Feature Classifier."""

from typing import List, Tuple
import chainer
from chainer import functions

from .mlp import SimpleLinear, LADL

CLASSIFIER_DICT = {
    "SimpleLinear": SimpleLinear,
    "LADL": LADL,
}


class ClassificationLayer(chainer.Chain):
    """Wrapper for Pooling Layer."""

    def __init__(self, n_class: int, classification_layer: str="SimpleLinear") -> None:
        """Initialize."""
        super(ClassificationLayer, self).__init__()

        with self.init_scope():
            self.cls = CLASSIFIER_DICT[classification_layer](n_class)

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward"""
        return self.cls(x)

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        return self(x)


class MultiHeadClassificationLayer(chainer.Chain):
    """Wrapper for Classification Layer, which output multiple Variables"""

    def __init__(
        self, n_classes: List[int]=[168, 11, 7],
        classification_layer: str="SimpleLinear", output_num: int=3
    ) -> None:
        """Initialize."""
        super(MultiHeadClassificationLayer, self).__init__()
        self.n_classes = n_classes
        self.output_num = len(n_classes)

        with self.init_scope():
            for i in range(self.output_num):
                setattr(
                    self, "cls{}".format(i + 1),
                    CLASSIFIER_DICT[classification_layer](self.n_classes[i]))

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward"""
        ret_list = [
            getattr(self, "cls{}".format(i + 1))(x) for i in range(self.output_num)]

        return functions.concat(ret_list, axis=1)

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        return self(x)


class TripleHeadClassificationLayer(MultiHeadClassificationLayer):
    """Wrapper for Pooling Layer, which output Triple Variables"""

    def __init__(
        self, n_classes: List[int]=[168, 11, 7], classification_layer: str="SimpleLinear"
    ) -> None:
        """Initialize."""
        super(TripleHeadClassificationLayer, self).__init__(
            n_classes, classification_layer, 3)


class MultiInOutClassificationLayer(chainer.Chain):
    """Wrapper for Classification Layer, which output multiple Variables"""

    def __init__(
        self, n_classes: List[int]=[168, 11, 7],
        classification_layer: str="SimpleLinear", output_num: int=3
    ) -> None:
        """Initialize."""
        super(MultiInOutClassificationLayer, self).__init__()
        self.n_classes = n_classes
        self.output_num = len(n_classes)

        with self.init_scope():
            for i in range(self.output_num):
                setattr(
                    self, "cls{}".format(i + 1),
                    CLASSIFIER_DICT[classification_layer](self.n_classes[i]))

    def __call__(self, xs: Tuple[chainer.Variable]) -> chainer.Variable:
        """Forward"""
        ret_list = [
            getattr(self, "cls{}".format(i + 1))(xs[i]) for i in range(self.output_num)]

        return functions.concat(ret_list, axis=1)

    def inference(self, xs: Tuple[chainer.Variable]) -> chainer.Variable:
        """Inference valid or test data."""
        return self(xs)


class TripleInOutClassificationLayer(MultiInOutClassificationLayer):
    """Wrapper for Pooling Layer, which output Triple Variables"""

    def __init__(
        self, n_classes: List[int]=[168, 11, 7], classification_layer: str="SimpleLinear"
    ) -> None:
        """Initialize."""
        super(TripleInOutClassificationLayer, self).__init__(
            n_classes, classification_layer, 3)
