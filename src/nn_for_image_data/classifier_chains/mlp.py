# -*- coding: utf-8 -*- #
"""Classifer Chains."""

import chainer
from chainer import functions, links, initializers


class SimpleLinear(chainer.Chain):
    """Chain to feed output of Extractor to Fully Connect."""

    def __init__(self, n_class: int) -> None:
        """Initialize."""
        super(SimpleLinear, self).__init__()
        with self.init_scope():
            self.tail_fc = links.Linear(
                None, n_class, initialW=initializers.Normal(scale=0.01))

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, input_dim) => return: (bs, n_class)
        return self.tail_fc(x)


class DropoutLinear(chainer.Chain):
    """Chain to feed output of Extractor to Dropout => FC."""

    def __init__(self, n_class: int, drop_rate: float=0.5):
        """Initialize."""
        super(DropoutLinear, self).__init__()
        with self.init_scope():
            self.tail_fc = links.Linear(
                None, n_class, initialW=initializers.Normal(scale=0.01))
        self.drop_rate = drop_rate

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, input_dim) => return: (bs, n_class)
        return self.tail_fc(functions.dropout(x, ratio=self.drop_rate))


class LADL(chainer.Chain):
    """Chain to feed output of Extractor to FC => ReLU => DO => FC."""

    def __init__(self, n_class: int, med_dim: int=1024, drop_rate: float=0.5) -> None:
        """Initialize."""
        super(LADL, self).__init__()
        with self.init_scope():
            self.med_fc = links.Linear(
                None, med_dim, initialW=initializers.Normal(scale=0.01))
            self.tail_fc = links.Linear(
                med_dim, n_class, initialW=initializers.Normal(scale=0.01))
        self.drop_rate = drop_rate

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, input_dim) => h: (bs, med_dim)
        h = functions.relu(self.med_fc(x))
        # # h: (bs, med_dim) => return: (bs, n_class)
        return self.tail_fc(functions.dropout(h, ratio=self.drop_rate))


class LBAL(chainer.Chain):
    """Chain to feed output of Extractor to FC => BN => ReLU => FC."""

    def __init__(self, n_class: int, med_dim: int=1024):
        """Initialize."""
        super(LBAL, self).__init__()
        with self.init_scope():
            self.med_fc = links.Linear(
                None, med_dim, initialW=initializers.Normal(scale=0.01))
            self.bn = links.BatchNormalization(med_dim)
            self.tail_fc = links.Linear(
                med_dim, n_class, initialW=initializers.Normal(scale=0.01))

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, input_dim) => h : (bs, med_dim)
        h = functions.relu(self.bn(self.med_fc(x)))
        # # h: (bs, med_dim) => return: (bs, n_class)
        return self.tail_fc(h)


class LBADL(chainer.Chain):
    """Chain to feed output of Extractor to FC => BN => ReLU => DO => FC."""

    def __init__(self, n_class: int, med_dim: int=1024, drop_rate: float=0.5):
        """Initialize."""
        super(LBADL, self).__init__()

        with self.init_scope():
            self.med_fc = links.Linear(
                None, med_dim, initialW=initializers.Normal(scale=0.01))
            self.bn = links.BatchNormalization(med_dim)
            self.tail_fc = links.Linear(
                med_dim, n_class, initialW=initializers.Normal(scale=0.01))
        self.drop_rate = drop_rate

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        # # x: (bs, input_dim) => h: (bs, med_dim)
        h = functions.relu(self.bn(self.med_fc(x)))
        # # h: (bs, med_dim) => return: (bs, n_class)
        return self.tail_fc(functions.dropout(h, ratio=self.drop_rate))
