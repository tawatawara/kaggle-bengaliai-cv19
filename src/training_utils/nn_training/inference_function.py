# -*- coding: utf-8 -*- #
"""Functions for inference."""

from time import time
from typing import Tuple, Union
import numpy as np

import chainer
from chainer import cuda, functions
from chainercv.links import PickableSequentialChain


def inference_test_data(
    model: Union[chainer.Chain, PickableSequentialChain],
    test_iter: chainer.iterators.MultiprocessIterator,
    gpu_device: int=-1
) -> Tuple[np.ndarray]:
    """Oridinary Inference."""
    test_pred_list = []
    test_label_list = []
    iter_num = 0
    epoch_test_start = time()

    while True:
        test_batch = test_iter.next()
        iter_num += 1
        print("\rtmp_iteration: {:0>5}".format(iter_num), end="")
        in_arrays = chainer.dataset.concat_examples(test_batch, gpu_device)

        # Forward the test data
        with chainer.no_backprop_mode() and chainer.using_config("train", False):
            prediction_test = model.inference(*in_arrays[:-1])
            test_pred_list.append(prediction_test)
            test_label_list.append(in_arrays[-1])
            prediction_test.unchain_backward()

        if test_iter.is_new_epoch:
            print(" => test end: {:.2f} sec".format(time() - epoch_test_start))
            test_iter.reset()
            break

    test_pred_all = cuda.to_cpu(functions.concat(test_pred_list, axis=0).data)
    test_label_all = cuda.to_cpu(functions.concat(test_label_list, axis=0).data)
    del test_pred_list
    del test_label_list
    return test_pred_all, test_label_all
