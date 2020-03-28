# -*- coding: utf-8 -*- #
"""Self-made extensions."""

from math import cos, pi
from typing import Union, Optional

import chainer


class CosineShiftByIter(chainer.training.extension.Extension):
    """
    Cosine Anealing.

    Fork from: https://github.com/takedarts/resnetfamily/blob/master/src/mylib/training/extensions/cosine_shift.py
    """

    def __init__(
        self, attr_name: str, epoch_per_cycle: int,
        max_value: Union[int, float], min_value: Union[int, float]=0,
        warm_up: Optional[str]=None, warm_up_epoch: int=0,
        optimizer: Optional[chainer.optimizer.GradientMethod]=None
    ) -> None:
        self.attr_name = attr_name
        self.max_value = max_value
        self.min_value = min_value
        self.min_value_for_warm_up = max(0, min_value)  # # for `negative` min_value
        self.epoch_per_cycle = epoch_per_cycle
        self.optimizer = optimizer
        self._last_value = None
        self.warm_up = warm_up
        self.warm_up_epoch = 0 if warm_up is None else warm_up_epoch
        if warm_up == "max":
            self._calc_next_value = [self._calc_next_value_by_max, self._calc_next_value_by_cosine]
        elif warm_up == "linear":
            self._calc_next_value = [self._calc_next_value_by_linear, self._calc_next_value_by_cosine]
        else:
            self._calc_next_value = [self._calc_next_value_by_cosine, self._calc_next_value_by_cosine]

    def initialize(self, trainer: chainer.training.trainer.Trainer) -> None:
        optimizer = self._get_optimizer(trainer)

        if self._last_value is not None:  # resuming from a snapshot
            setattr(optimizer, self.attr_name, self._last_value)
        else:
            setattr(optimizer, self.attr_name, self.max_value)
            self._last_value = self.max_value

    def __call__(self, trainer: chainer.training.trainer.Trainer) -> None:
        self._update_value(trainer)

    def _update_value(self, trainer: chainer.training.trainer.Trainer) -> None:
        optimizer = self._get_optimizer(trainer)
        epoch = trainer.updater.epoch
        epoch_detail = trainer.updater.epoch_detail

        next_value = self._calc_next_value[epoch >= self.warm_up_epoch](epoch, epoch_detail)
        setattr(optimizer, self.attr_name, next_value)
        self._last_value = next_value

    def _calc_next_value_by_max(self, epoch: int, epoch_detail: float) -> Union[int, float]:
        return self.max_value

    def _calc_next_value_by_linear(self, epoch: int, epoch_detail: float) -> Union[int, float]:
        return self.min_value_for_warm_up + (self.max_value - self.min_value_for_warm_up) * epoch_detail / self.warm_up_epoch

    def _calc_next_value_by_cosine(self, epoch: int, epoch_detail: float) -> Union[int, float]:
        progress = ((epoch - self.warm_up_epoch) % self.epoch_per_cycle + (epoch_detail - epoch)) / self.epoch_per_cycle
        return self.min_value + (self.max_value - self.min_value) * (1 + cos(progress * pi)) / 2

    def _get_optimizer(self, trainer: chainer.training.trainer.Trainer) -> chainer.optimizer.GradientMethod:
        return self.optimizer or trainer.updater.get_optimizer('main')
