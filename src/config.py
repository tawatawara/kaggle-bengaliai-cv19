# -*- coding: utf-8 -*- #
"""Config for this competition."""

from pathlib import Path

COMPETITION_NAME = "bengaliai-cv19"

ROOT = Path(".").absolute().parents[0]
INPUT_ROOT = ROOT / "input"
OUTPUT_ROOT = ROOT / "output"
RAW_DATA = INPUT_ROOT / COMPETITION_NAME
PROC_DATA = ROOT / "processed_data"
TRAIN_IMAGES_DIR = PROC_DATA / "train"
TEST_IMAGES_DIR = PROC_DATA / "test"

FOLD_NUM = 4
RANDAM_SEED = 39

PARUET_FILE_NUM = 4
ORIGINAL_IMAGE_HIGHT = 137
ORIGINAL_IMAGE_WIDTH = 236

TRAINING_IMAGE_HIGHT = 128
TRAINING_IMAGE_WIDTH = 224


TRAIN_IMAGE_MEAN = [241.47665114988533, ]
TRAIN_IMAGE_STD = [16.781429968447544, ]

TRAIN_IMAGE_MEAN_NORM = [0.946967259411315, ]
TRAIN_IMAGE_MEAN_STD = [0.06580952928802959, ]

N_CLASSES = [168, 11, 7]
COMP_INDEXS = [sum(N_CLASSES[:i]) for i in range(len(N_CLASSES) + 1)]
COMP_NAMES = ["grapheme_root", "vowel_diacritic", "consonant_diacritic"]

if __name__ == "__main__":
    print("COMPETITION_NAME:", COMPETITION_NAME)
    print("ROOT_PATH:", ROOT)
    print("INPUT_ROOT:", INPUT_ROOT)
    print("OUTPUT_ROOT:", OUTPUT_ROOT)
    print("RAW_DATA:", RAW_DATA)
    print("PROC_DATA:", PROC_DATA)
