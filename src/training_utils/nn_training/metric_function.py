# -*- coding: utf-8 -*- #
"""metric functions."""
from typing import List, Dict
import numpy as np

import chainer
from chainer import functions
from chainer import cuda


class FocalLoss:
    r"""
    class for Focal loss.

    calculates focal loss **for each class**, **sum up** them along class-axis, and **average** them along sample-axis.
    Take data point x and its logit y = model(x),
    using prob p = (p_0, ..., p_C)^T = prob_func(y) and label t,
    focal loss for each class i caluculated by:

        loss_{i}(p, t) = - \alpha' + (1 - p'_i) ** \gamma * ln(p'_i),

    where
        \alpha' = { \alpha (t_i = 1)
                  { 1 - \alpha (t_i = 0)
         p'_i   = { p_i (t_i = 1)
                = ( 1 - p_i (t_i = 0)
    """

    def __init__(
        self, alpha: float=0.25, gamma: float=2, prob_func: str="sigmoid", label_arr_dim: int=2
    ):
        """Initialize."""
        self.alpha = alpha
        self.gamma = gamma
        self.prob_func = {
            "sigmoid": functions.sigmoid,
            "softmax": functions.softmax}[prob_func]
        self.reshape_func = {2: lambda t, n_class: t, 1: self._convert_label}[label_arr_dim]

    def __call__(self, y_pred: chainer.Variable, t: np.ndarray, epsilon: float=1e-31):
        r"""
        Forward.

        p_dash = t * p + (1 - t) * (1 - p) = (1 - t) + (2 * t - 1) * p
        alpha_dash = t * \alpha + (1 - t) * (1 -  \alpha) = (1 - t) + (2 * t - 1) * \alpha
        """
        bs, n_class = y_pred.shape
        t = self.reshape_func(t, n_class)

        p_dash = functions.clip(
            (1 - t) + (2 * t - 1) * self.prob_func(y_pred), epsilon, 1 - epsilon)
        alpha_dash = (1 - t) + (2 * t - 1) * self.alpha
        # # [y_pred: (bs, n_class), t: (bs: n_class)] => loss_by_sample_x_class: (bs, n_class)
        loss_by_sample_x_class = - alpha_dash * (1 - p_dash) ** self.gamma * functions.log(p_dash)
        # # loss_by_sample_x_class: (bs, n_class) => loss_by_sample: (bs, )
        loss_by_sample = functions.sum(loss_by_sample_x_class, axis=1)
        # # loss_by_sample: (bs,) => loss: (1, )
        return functions.mean(loss_by_sample)

    def _convert_label(self, t: np.ndarray, n_class: int):
        """Reshape Label Array."""
        xp = cuda.get_array_module(t)
        return xp.identity(n_class, dtype="i")[t]


class FocalSoftMaxCrossEntropy:
    """Calculate custom softmax cross entropy loss with the idea of focal loss."""

    def __init__(self, alpha: float=1, gamma: float=2) -> None:
        """Initialize."""
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, y_pred: chainer.Variable, t: np.ndarray, epsilon: float=1e-31) -> chainer.Variable:
        """Forward."""
        xp = cuda.get_array_module(t)
        p = functions.clip(functions.softmax(y_pred), epsilon, 1 - epsilon)[xp.arange(t.shape[0]), t]
        # loss_by_sample = - self.alpha * (1 - p) ** self.gamma * functions.log(p)
        loss_by_sample = - (1 - p) ** self.gamma * functions.log(p)
        return functions.mean(loss_by_sample)


class WeightedLoss:
    """
    Calculate loss for each conmponent and weighting them.

    l1 = softmax(p1, t1)
    l2 = softmax(p2, t2)
    l3 = softmax(p3, t3)

    return w1 * l1 + w2 * l2 + w3 * l3.
    """

    def __init__(
        self, n_classes: List[int], weights: List[float],
        loss_func: str="softmax_cross_entropy", loss_kwargs: Dict=lambda: dict()
    ) -> None:
        """Initialize."""
        self.n_classes = n_classes
        self.n_component = len(n_classes)
        self.comp_indexs = [sum(self.n_classes[:i]) for i in range(self.n_component + 1)]
        self.weights = np.array(weights, dtype="float32")
        self.weights /= self.weights.sum()
        if loss_func == "softmax_cross_entropy":
            self.loss_func = functions.softmax_cross_entropy
        else:
            self.loss_func = {
                "focal_loss": FocalLoss,
                "focal_softmax_cross_entropy": FocalSoftMaxCrossEntropy
            }[loss_func](**loss_kwargs)

    def __call__(self, y_pred: chainer.Variable, t: np.ndarray) -> chainer.Variable:
        """Forward."""
        loss_each = functions.stack([
            self.loss_func(
                y_pred[:, self.comp_indexs[i]: self.comp_indexs[i + 1]], t[:, i]
            ) * self.weights[i]
            for i in range(self.n_component)])

        return functions.sum(loss_each)


class SelectedSoftmaxCrossEntropy:
    """This class is only for validation. To see each conponents loss."""

    def __init__(self, n_classes: List[int], component_id=0) -> None:
        """Initialize."""
        self.n_classes = n_classes
        self.n_component = len(n_classes)
        self.comp_indexs = [sum(self.n_classes[:i]) for i in range(self.n_component + 1)]
        self.comp_id = component_id

    def __call__(self, y_pred: chainer.Variable, t: np.ndarray) -> chainer.Variable:
        """Forward."""
        return functions.softmax_cross_entropy(
            y_pred[:, self.comp_indexs[self.comp_id]: self.comp_indexs[self.comp_id + 1]],
            t[:, self.comp_id])
