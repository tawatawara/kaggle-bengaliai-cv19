# -*- coding: utf-8 -*- #
"""script for train."""
import os
import shutil
import gc
from pathlib import Path, PosixPath
from argparse import ArgumentParser

import yaml
import numpy as np
import pandas as pd
import matplotlib

from sklearn.metrics import recall_score
from chainer import serializers, datasets

from competition_utils import utils
from nn_for_image_data import backborn_chains, global_pooling_chains, classifer_chains
from training_utils import nn_training

import config
matplotlib.use('Agg')


def argparse():
    """Parse Comndline Args."""
    usage_msg = """
\n  python {0} --setting_file_path <str> --output_file_path <str>\n
""".format(__file__,)
    parser = ArgumentParser(
        prog="nn_train.py", usage=usage_msg)

    parser.add_argument("-s", "--setting_file_path", dest="setting_file_path", default="./settings.yml")
    parser.add_argument("-o", "--output_path", dest="output_path", default="")
    argvs = parser.parse_args()
    return argvs


def train(settings: dict, output_path: PosixPath):
    """Main."""
    gpu_num = len(settings["gpu_devices"])
    # # make dataset
    # # # read meta info.
    train_df = pd.read_csv(config.PROC_DATA / "train_add-{}fold-index.csv".format(settings["n_folds"]))

    # # # make label arr
    train_labels_arr = train_df[config.COMP_NAMES].values.astype("i")

    # # # make train set
    if settings["val_fold"] != -1:
        train_dataset = datasets.LabeledImageDataset(
            pairs=list(zip(
                (train_df[train_df["fold"] != settings["val_fold"]]["image_id"] + ".png").tolist(),
                train_labels_arr[train_df["fold"] != settings["val_fold"], ...])),
            root=config.TRAIN_IMAGES_DIR.as_posix())
    else:
        train_dataset = datasets.LabeledImageDataset(
            pairs=list(zip(
                (train_df["image_id"] + ".png").tolist(), train_labels_arr)),
            root=config.TRAIN_IMAGES_DIR.as_posix())

    train_dataset = datasets.TransformDataset(
        train_dataset, nn_training.ImageTransformer(settings["training_transforms"]))

    if gpu_num > 1:
        # # if using multi-gpu, split train set into gpu_num.
        train_sub_dataset_list = []
        total_size = len(train_dataset)
        subset_size = (total_size + gpu_num - 1) // gpu_num
        np.random.seed(1086)
        random_order = np.random.permutation(len(train_dataset))
        for i in range(gpu_num):
            start_idx = min(i * subset_size, total_size - subset_size)
            end_idx = min((i + 1) * subset_size, total_size)
            print(i, start_idx, end_idx)
            train_sub_dataset_list.append(
                datasets.SubDataset(
                    train_dataset, start=start_idx, finish=end_idx, order=random_order)
            )
        train_dataset = train_sub_dataset_list

        for i, subset in enumerate(train_dataset):
            print("subset{}: {}".format(i, len(subset)))

    # # # # validation set
    if settings["val_fold"] != -1:
        val_dataset = datasets.LabeledImageDataset(
            pairs=list(zip(
                (train_df[train_df["fold"] == settings["val_fold"]]["image_id"] + ".png").tolist(),
                train_labels_arr[train_df["fold"] == settings["val_fold"], ...])),
            root=config.TRAIN_IMAGES_DIR.as_posix())
    else:
        # # if train models using all train data, calc loss for all data at the evaluation step.
        val_dataset = datasets.LabeledImageDataset(
            pairs=list(zip(
                (train_df["image_id"] + ".png").tolist(), train_labels_arr)),
            root=config.TRAIN_IMAGES_DIR.as_posix())

    val_dataset = datasets.TransformDataset(
        val_dataset, nn_training.ImageTransformer(settings["inference_transforms"]))

    print("[make dataset] train: {}, val: {}".format(len(train_dataset), len(val_dataset)))

    # # initialize model.
    model = nn_training.ImageClassificationModel(
        extractor=getattr(
            backborn_chains, settings["backborn_class"])(**settings["backborn_kwargs"]),
        global_pooling=None if settings["pooling_class"] is None else getattr(
            global_pooling_chains, settings["pooling_class"])(**settings["pooling_kwargs"]),
        classifier=getattr(
            classifer_chains, settings["head_class"])(**settings["head_kwargs"])
    )
    model.name = settings["model_name"]

    # # set training wrapper.
    train_model = nn_training.CustomClassifier(
        predictor=model,
        lossfun=getattr(nn_training, settings["loss_function"][0])(**settings["loss_function"][1]),
        evalfun_dict={
            "SCE_{}".format(i): getattr(nn_training, name)(**param)
            for i, (name, param) in enumerate(settings["eval_functions"])})

    settings["eval_func_names"] = ["SCE_{}".format(i) for i in range(len(settings["eval_functions"]))]

    gc.collect()
    # # training.
    # # # create trainer.
    utils.set_random_seed(settings["seed"])
    trainer = nn_training.create_trainer(
        settings, output_path.as_posix(), train_model, train_dataset, val_dataset)
    trainer.run()

    # # # save model of last epoch,
    model = trainer.updater.get_optimizer('main').target.predictor
    serializers.save_npz(
        output_path / "model_snapshot_last_epoch.npz", model)

    del trainer
    del train_model
    gc.collect()

    # # inference validation data by the model of last epoch.
    _, val_iter, _ = nn_training.create_iterator(settings, None, val_dataset, None)
    val_pred, val_label = nn_training.inference_test_data(model, val_iter, gpu_device=settings["gpu_devices"][0])
    np.save(output_path / "val_pred_arr_fold{}".format(settings["val_fold"]), val_pred)

    # # calc validation score
    score_list = [[] for i in range(2)]

    for i in range(len(config.N_CLASSES)):
        y_pred_subset = val_pred[:, config.COMP_INDEXS[i]:config.COMP_INDEXS[i + 1]].argmax(axis=1)
        y_true_subset = val_label[:, i]
        score_list[0].append(
            recall_score(y_true_subset, y_pred_subset, average='macro', zero_division=0))
        score_list[1].append(
            recall_score(y_true_subset, y_pred_subset, average='macro', zero_division=1))
    score_list[0].append(np.average(score_list[0], weights=[2, 1, 1]))
    score_list[1].append(np.average(score_list[1], weights=[2, 1, 1]))

    score_df = pd.DataFrame(
        score_list, columns=config.COMP_NAMES + ["score"])

    print(score_df)
    score_df.to_csv(output_path / "score.csv", index=False)


def main():
    """Main."""
    argvs = argparse()
    setting_file_path = Path(argvs.setting_file_path)

    with open(setting_file_path.as_posix(), "r") as fr:
        settings = yaml.safe_load(fr)

    if argvs.output_path == "":
        output_path = config.OUTPUT_ROOT / "{}_{}".format(utils.get_timestamp(), settings["model_name"])
    else:
        output_path = Path(argvs.output_path).resolve()

    if os.path.isdir(output_path):
        print("Directory `{}` already exists. ".format(output_path))
        print("You must remove it or specify the other directory.")
        quit()
    os.mkdir(output_path)
    shutil.copyfile(Path(".") / "nn_train.py", output_path / "nn_train.py")
    shutil.copyfile(setting_file_path, output_path / "settings.yml")

    train(settings, output_path)


if __name__ == "__main__":
    main()
