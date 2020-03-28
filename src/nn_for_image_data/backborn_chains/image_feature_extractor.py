# -*- coding: utf-8 -*- #
"""Wrapper of Feature Extractor."""

from pathlib import Path
from typing import Optional, List

import chainer
from chainercv import links

from .seresnext_conv1_mean import SEResNeXt50Conv1MeanTo1ch, SEResNeXt101Conv1MeanTo1ch

import sys

BACKBORN_DICT = {
    "SEResNeXt50": links.SEResNeXt50,
    "SEResNeXt101": links.SEResNeXt101,
    "SEResNeXt50Conv1MeanTo1ch": SEResNeXt50Conv1MeanTo1ch,
    "SEResNeXt101Conv1MeanTo1ch": SEResNeXt101Conv1MeanTo1ch,
}

# INPUT_ROOT = Path(sys.path[0]).parents[2] / "input"
INPUT_ROOT = Path(sys.path[0]).parents[0] / "input"

SERESNEXT_PATH = INPUT_ROOT / "chainercv-seresnext"
SERESNEXT50_PATH = (SERESNEXT_PATH / "se_resnext50_imagenet_converted_2018_06_28.npz").as_posix()
SERESNEXT101_PATH = (SERESNEXT_PATH / "se_resnext101_imagenet_converted_2018_06_28.npz").as_posix()

USUAL_PRETRAINED_PATH_DICT = {
    "SEResNeXt50": SERESNEXT50_PATH,
    "SEResNeXt101": SERESNEXT101_PATH,
    "SEResNeXt50Conv1MeanTo1ch": SERESNEXT50_PATH,
    "SEResNeXt101Conv1MeanTo1ch": SERESNEXT101_PATH,
}


class ImageFeatureExtractor(chainer.Chain):
    """image feture extractor based on pretrained model."""

    def __init__(
        self, backborn_model: str,
        pretrained_model_path: Optional[str]="usual", extract_layers: List[str]=["res5"]
    ) -> None:
        """Initialze."""
        super(ImageFeatureExtractor, self).__init__()

        if hasattr(links, backborn_model):
            backborn_model_class = getattr(links, backborn_model)
        else:
            backborn_model_class = eval(backborn_model)

        if pretrained_model_path is not None and pretrained_model_path == "usual":
            pretrained_model_path = USUAL_PRETRAINED_PATH_DICT[backborn_model]

        with self.init_scope():
            self.extractor = backborn_model_class(pretrained_model=pretrained_model_path)
        self.extractor._pick = extract_layers
        self.extractor.remove_unused()
        # print(self._children)
        # print(self.extractor.layer_names)

    def __call__(self, x: chainer.Variable) -> chainer.Variable:
        """Forward."""
        return self.extractor(x)

    def inference(self, x: chainer.Variable) -> chainer.Variable:
        """Inference valid or test data."""
        return self(x)
