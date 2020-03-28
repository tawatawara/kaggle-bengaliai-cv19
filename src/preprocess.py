# -*- coding: utf-8 -*- #
"""Preprocess Meta Information"""

import os
import gc
from PIL import Image
from pathlib import PosixPath
from argparse import ArgumentParser

import pandas as pd
from competition_utils import utils
import config


def argparse():
    """Parse Comndline Args."""
    usage_msg = """
\n  python {0} [-tr] [-te]\n
""".format(__file__,)
    parser = ArgumentParser(
        prog="nn_train.py", usage=usage_msg)

    parser.add_argument('-tr', "--train", dest="train", action='store_true')
    parser.add_argument('-te', "--test", dest="test", action='store_true')
    argvs = parser.parse_args()
    return argvs


def convert_parquet_to_images(
    parquet_file_path: PosixPath, image_dir_path: PosixPath
) -> bool:
    """Convert a parquet file to image files and save them"""
    df = pd.read_parquet(parquet_file_path)
    image_ids = df.image_id.values
    arrs = df.iloc[:, 1:].values
    del df
    gc.collect()

    for i, image_id in enumerate(image_ids):
        Image.fromarray(
            arrs[i, :].reshape(config.ORIGINAL_IMAGE_HIGHT, config.ORIGINAL_IMAGE_WIDTH)
        ).save(image_dir_path / "{}.png".format(image_id))
    del arrs
    del image_ids
    gc.collect()


def preprocess_parquet_files(process_train: bool, process_test: bool) -> None:
    """Read parquet files and convert them into images."""
    if process_train:
        # # train
        print("preprocess train parquet files")
        if os.path.isdir(config.TRAIN_IMAGES_DIR):
            print("train images dir already exists!")
            pass
        else:
            os.mkdir(config.TRAIN_IMAGES_DIR)
            for i in range(config.PARUET_FILE_NUM):
                pqt_file_name = "train_image_data_{}.parquet".format(i)
                parquet_file_path = config.RAW_DATA / pqt_file_name
                with utils.timer("convert {} to png files.".format(pqt_file_name)):
                    convert_parquet_to_images(parquet_file_path, config.TRAIN_IMAGES_DIR)

    if process_test:
        # # test
        print("preprocess test parquet files")
        if os.path.isdir(config.TEST_IMAGES_DIR):
            print("test images dir already exists!")
            pass
        else:
            os.mkdir(config.TEST_IMAGES_DIR)
            for i in range(config.PARUET_FILE_NUM):
                pqt_file_name = "test_image_data_{}.parquet".format(i)
                parquet_file_path = config.RAW_DATA / pqt_file_name
                with utils.timer("convert {} to png files.".format(pqt_file_name)):
                    convert_parquet_to_images(parquet_file_path, config.TEST_IMAGES_DIR)


def preprocess_meta_info_files(process_train: bool, process_test: bool) -> None:
    """Preprocess Train and Test Meta Info."""
    if process_train:
        with utils.timer("preprocess train meta file"):
            train = pd.read_csv(config.RAW_DATA / "train.csv")
            # # K-fold split.
            train["character_id"] = train.apply(
                lambda row: "{:0>3}_{:0>2}_{}".format(
                    row["grapheme_root"], row["vowel_diacritic"], row["consonant_diacritic"]), axis=1)

            labels_arr = pd.get_dummies(
                train[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]],
                columns=["grapheme_root", "vowel_diacritic", "consonant_diacritic"]).values

            train["fold"] = -1
            for fold_id, (train_idx, valid_idx) in enumerate(
                utils.multi_label_stratified_group_k_fold(
                    train.character_id.values, labels_arr, config.FOLD_NUM, config.RANDAM_SEED)
            ):
                train.loc[valid_idx, "fold"] = fold_id

            train.to_csv(config.PROC_DATA / "train_add-{}fold-index.csv".format(config.FOLD_NUM), index=False)

    if process_test:
        with utils.timer("preprocess test meta file"):
            test = pd.read_csv(config.RAW_DATA / "test.csv")
            test_proc = pd.DataFrame({"image_id": test.image_id.drop_duplicates().values})
            test_proc["grapheme_root"] = 0
            test_proc["vowel_diacritic"] = 0
            test_proc["consonant_diacritic"] = 0
            test_proc.to_csv(config.PROC_DATA / "test_reshaped.csv", index=False)


def main():
    """Main."""
    argvs = argparse()
    preprocess_parquet_files(argvs.train, argvs.test)
    preprocess_meta_info_files(argvs.train, argvs.test)


if __name__ == "__main__":
    main()
