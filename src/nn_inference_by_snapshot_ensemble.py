# -*- coding: utf-8 -*- #
"""script for inference validation and test."""
import gc
import os
import shutil
from pathlib import Path, PosixPath
from argparse import ArgumentParser

import yaml
import numpy as np
import pandas as pd

from chainer import serializers, datasets, functions

from competition_utils import utils
from nn_for_image_data import backbone_chains, global_pooling_chains, classifier_chains
from training_utils import nn_training

import config


def argparse():
    """Parse Comndline Args."""
    usage_msg = """
\n  python {0} --trained_path <str> --output_path <str> --gpu_device <int> --batch_size <int>\n
""".format(__file__,)
    parser = ArgumentParser(prog="nn_inference_by_snapshot_ensemble.py", usage=usage_msg)

    parser.add_argument("-t", "--trained_path", dest="trained_path", required=True)
    parser.add_argument("-o", "--output_path", dest="output_path", default="")
    parser.add_argument("-g", "--gpu_device", dest="gpu_device", default=-1, type=int)
    parser.add_argument("-bs", "--batch_size", dest="batch_size", default=64, type=int)
    argvs = parser.parse_args()
    return argvs


def init_model(settings):
    model = nn_training.ImageClassificationModel(
        extractor=getattr(
            backbone_chains, settings["backbone_class"])(**settings["backbone_kwargs"]),
        global_pooling=None if settings["pooling_class"] is None else getattr(
            global_pooling_chains, settings["pooling_class"])(**settings["pooling_kwargs"]),
        classifier=getattr(
            classifier_chains, settings["head_class"])(**settings["head_kwargs"])
    )
    return model


def inference_by_snapshot_ensemble(
    trained_path: PosixPath, output_path: PosixPath, gpu_device: int=-1, batch_size: int=64
):
    """Inference function for kernel."""
    # # read settings from training outputs directory.
    with open((trained_path / "settings.yml").as_posix(), "r") as fr:
        settings = yaml.safe_load(fr)

    # # make dataset
    # # # test set
    with utils.timer("make test dataset"):
        test_df = pd.read_csv(config.PROC_DATA / "test_reshaped.csv")
        sample_sub = pd.read_csv(config.RAW_DATA / "sample_submission.csv")

        # # # # make chainer dataset
        test_dataset = datasets.LabeledImageDataset(
            pairs=list(zip((test_df["image_id"] + ".png").tolist(), ([-1] * len(test_df)))),
            root=config.TEST_IMAGES_DIR.as_posix())
        # # # # set transform
        test_dataset = datasets.TransformDataset(
            test_dataset, nn_training.ImageTransformer(settings["inference_transforms"]))

    # # # prepare model paths
    model_path_list = []
    model_weight = []
    for epoch_of_model in range(
        settings["epoch_per_cycle"], settings["max_epoch"] + 1, settings["epoch_per_cycle"]
    ):
        model_path = trained_path / "model_snapshot_{}.npz".format(epoch_of_model)
        if os.path.isfile(model_path):
            model_path_list.append(model_path)
            model_weight.append(1)

    if len(model_path_list) == 0:
        model_path_list.append(trained_path / "model_snapshot_last_epoch.npz")
        model_weight.append(1)
    print("[using models]")
    print(model_path_list)

    # # # prepare preds numpy.ndarray of shape: (n_model, n_test, n_class)
    test_preds_arr = np.zeros(
        (len(model_path_list), len(test_df), sum(config.N_CLASSES)), dtype="f")

    # # inference
    with utils.timer("inference test set"):
        for idx, model_path in enumerate(model_path_list):
            # # # create iterator.
            test_iter = nn_training.create_iterator(settings, None, None, test_dataset)[-1]
            # # # init and load model
            model = init_model(settings)
            serializers.load_npz(model_path, model)
            # # # move model to gpu
            model.to_gpu(gpu_device)
            # # # inference
            test_preds_arr[idx] = nn_training.inference_test_data(model, test_iter, gpu_device=gpu_device)[0]
            del test_iter
            del model
            gc.collect()
        del test_dataset

    np.save(output_path / "test_all_preds_arr_fold{}".format(settings["val_fold"]), test_preds_arr)

    # # ensemble (weighted averaging)
    with utils.timer("snapshot ensemble"):
        # # # convert logits to probs
        for i in range(len(config.N_CLASSES)):
            test_preds_arr[..., config.COMP_INDEXS[i]:config.COMP_INDEXS[i + 1]] =\
                functions.softmax(test_preds_arr[..., config.COMP_INDEXS[i]:config.COMP_INDEXS[i + 1]]).data

        test_pred = np.average(test_preds_arr, axis=0, weights=model_weight)
        np.save(output_path / "test_pred_arr_fold{}".format(settings["val_fold"]), test_pred)

    with utils.timer("make submission"):
        # # convert prob to pred id
        for i, c_name in enumerate(config.COMP_NAMES):
            test_pred_subset = test_pred[:, config.COMP_INDEXS[i]:config.COMP_INDEXS[i + 1]].argmax(axis=1)
            test_df[c_name] = test_pred_subset

        del test_pred_subset
        del test_pred
        gc.collect()

        # # # reshape test_df to submisson format.
        melt_df = pd.melt(test_df, id_vars="image_id", value_vars=config.COMP_NAMES, value_name="target")
        melt_df["row_id"] = melt_df["image_id"] + "_" + melt_df["variable"]

        submission_df = pd.merge(
            sample_sub[["row_id"]], melt_df[["row_id", "target"]], on="row_id", how="left")

        submission_df.to_csv(output_path / "submission.csv", index=False)


def main():
    """Main."""
    argvs = argparse()
    trained_path = Path(argvs.trained_path).resolve()
    output_path = trained_path

    if argvs.output_path != "":
        output_path = Path(argvs.output_path).resolve()
        if os.path.isdir(output_path):
            print("Directory `{}` already exists. ".format(output_path))
            print("You must remove it or specify the other directory.")
            quit()

        os.mkdir(output_path)

    shutil.copyfile(
        Path(".") / "nn_inference_by_snapshot_ensemble.py",
        output_path / "nn_inference_by_snapshot_ensemble.py")

    inference_by_snapshot_ensemble(
        trained_path, output_path, argvs.gpu_device, argvs.batch_size)


if __name__ == "__main__":
    main()
