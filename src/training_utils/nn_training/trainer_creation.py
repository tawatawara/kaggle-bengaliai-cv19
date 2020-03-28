# -*- coding: utf-8 -*- #
"""functions for training."""
from typing import Tuple, Dict, Union, Optional

import chainer
from chainer import datasets, links, iterators, optimizers, training, cuda

from .my_extension import CosineShiftByIter


def create_trainer(
    settings: Dict, out_dir: str,
    train_model: links.Classifier,
    train_dataset: Union[datasets.LabeledImageDataset, datasets.TupleDataset, None],
    val_dataset: Union[datasets.LabeledImageDataset, datasets.TupleDataset, None],
    test_dataset: Union[datasets.LabeledImageDataset, datasets.TupleDataset, None]=None,
) -> training.trainer.Trainer:
    """Create Trainer.

    Arguments
    ----------
    settings: Dict
        arguments for creating trainer
    out_dir: Str
        directory path for saving training output such as trained models, training logs
    model: chainer.Chain
        nn model class
    train_dataset, val_dataset, test_dataset: chainer.Dataset
    """
    # # set optimizer.
    optimizer = set_optimizer(settings, train_model)

    # # prepare dataset iterator
    train_iter, val_iter, test_iter = create_iterator(
        settings, train_dataset, val_dataset, test_dataset)

    # # make updater.
    gpu_ids = settings["gpu_devices"]
    if len(gpu_ids) < 2:
        updater = training.StandardUpdater(
            train_iter, optimizer, device=gpu_ids[0])
    else:
        master_gpu_id = gpu_ids[0]
        cuda.get_device_from_id(master_gpu_id)
        devices_dict = {"main": master_gpu_id}
        devices_dict.update(
            {"sub{}".format(i + 1): gid for i, gid in enumerate(gpu_ids[1:])})
        updater = training.updaters.MultiprocessParallelUpdater(
            iterators=train_iter, optimizer=optimizer, devices=devices_dict)

    # # make trainer.
    if settings["use_early_stopping"]:
        stop_trigger = training.triggers.EarlyStoppingTrigger(
            check_trigger=tuple(settings["es_check_trigger"]),
            monitor='val/main/{}'.format(settings["eval_functions"][0]), mode="min",
            patients=settings["patience"], max_trigger=(settings["max_epoch"], 'epoch'), verbose=True)
    else:
        stop_trigger = (settings["max_epoch"], "epoch")
    trainer = training.trainer.Trainer(updater, stop_trigger, out=out_dir)

    # # set extentions (eval, log, plot).
    trainer = set_extensions(settings, trainer, val_iter, test_iter)

    return trainer


def set_optimizer(settings: Dict, train_model: links.Classifier) -> chainer.optimizer.GradientMethod:
    """Set optimizer to model."""
    # # set optimizer.
    opt_class = getattr(optimizers, settings["optimizer"])
    if settings["optimizer"] != "Adam":
        optimizer = opt_class(lr=settings["learning_rate"], momentum=settings["momentum"])
        optimizer.setup(train_model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(settings["weight_decay_rate"]))
    else:
        optimizer = opt_class(
            alpha=settings["learning_rate"], weight_decay_rate=settings["weight_decay_rate"])
        optimizer.setup(train_model)

    return optimizer


def create_iterator(
    settings: Dict,
    train_dataset: Union[datasets.LabeledImageDataset, datasets.TupleDataset, None]=None,
    val_dataset: Union[datasets.LabeledImageDataset, datasets.TupleDataset, None]=None,
    test_dataset: Union[datasets.LabeledImageDataset, datasets.TupleDataset, None]=None
) -> Tuple[Optional[chainer.iterators.MultiprocessIterator]]:
    """Create dataset iterator."""
    gpu_num = len(settings["gpu_devices"])
    if train_dataset is None:
        train_iter = None
    else:
        if gpu_num == 1:
            train_iter = iterators.MultiprocessIterator(
                train_dataset, settings["batch_size"], n_processes=settings["n_processes"])
        else:
            assert gpu_num == len(train_dataset),\
                " gpu num: {} != dataset num: {}".format(gpu_num, len(train_dataset))
            train_iter = [
                iterators.MultiprocessIterator(
                    sub_dataset, settings["batch_size"], n_processes=min(2, settings["n_processes"] // gpu_num))
                for sub_dataset in train_dataset]

    val_iter = None if val_dataset is None else \
        iterators.MultiprocessIterator(
            val_dataset, settings["batch_size"], repeat=False, shuffle=False, n_processes=settings["n_processes"])

    test_iter = None if test_dataset is None else \
        iterators.MultiprocessIterator(
            test_dataset, settings["batch_size"], repeat=False, shuffle=False, n_processes=settings["n_processes"])

    return train_iter, val_iter, test_iter


def set_extensions(
    settings: Dict,
    trainer: training.trainer.Trainer,
    val_iter: Optional[chainer.iterators.MultiprocessIterator],
    test_iter: Optional[chainer.iterators.MultiprocessIterator]=None
) -> training.trainer.Trainer:
    """Set extentions."""
    log_trigger = tuple(settings["log_trigger"])
    eval_trigger = tuple(settings["eval_trigger"])
    print("[log freq]", log_trigger)
    logging_attributes = [log_trigger[1]]

    loss_prefix = ["main", "val/main"]
    eval_prefix = ["val/main"]
    lr_attr_name = "alpha" if settings["optimizer"] == "Adam" else "lr"

    if test_iter is not None:
        loss_prefix.append("test/main")
        eval_prefix.append("test/main")

    loss_attr = ["{}/loss".format(lp) for lp in loss_prefix]
    logging_attributes.extend(loss_attr)
    if val_iter is not None:
        eval_attr = ["{}/{}".format(ep, ef) for ep in eval_prefix for ef in settings["eval_func_names"]]
        logging_attributes.extend(eval_attr)

    # if settings["optimizer"] == "Adam":
    #     logging_attributes.extend(["lr", lr_attr_name, "elapsed_time"])
    # else:
    #     logging_attributes.extend([lr_attr_name, "elapsed_time"])
    logging_attributes.extend([lr_attr_name, "elapsed_time"])

    # # lr scheduler.
    if settings["learning_schedule"] == "cosine":
        # # # cosine anealing.
        trainer.extend(
            CosineShiftByIter(
                attr_name=lr_attr_name, epoch_per_cycle=settings["epoch_per_cycle"],
                max_value=settings["learning_rate"], min_value=settings["learning_rate_min"],
                warm_up=settings["warm_up"], warm_up_epoch=settings["warm_up_epoch"])
        )

    # # evaluator.
    if val_iter is not None:
        eval_target = trainer.updater.get_optimizer('main').target
        trainer.extend(
            training.extensions.Evaluator(
                val_iter, eval_target, device=settings["gpu_devices"][0], eval_func=eval_target.evaluate),
            name="val", trigger=eval_trigger)

    if test_iter is not None:
        trainer.extend(
            training.extensions.Evaluator(
                test_iter, eval_target, device=settings["gpu_devices"][0], eval_func=eval_target.evaluate),
            name="test", trigger=eval_trigger)

    # #  log.
    trainer.extend(training.extensions.observe_lr(), trigger=log_trigger)
    if settings["optimizer"] == "Adam":
        trainer.extend(
            training.extensions.observe_lr(observation_key=lr_attr_name), trigger=log_trigger)

    trainer.extend(
        training.extensions.LogReport(logging_attributes, trigger=log_trigger), trigger=log_trigger)
    # # standard output.
    trainer.extend(training.extensions.PrintReport(logging_attributes), trigger=log_trigger)
    # trainer.extend(training.extensions.ProgressBar(update_interval=50))

    # # # plots.
    trainer.extend(
        training.extensions.PlotReport(
            ["{}/loss".format(lp) for lp in loss_prefix], "epoch", file_name="loss.png"),
        trigger=log_trigger)

    for ef in settings["eval_func_names"]:
        trainer.extend(
            training.extensions.PlotReport(
                ["{}/{}".format(ep, ef) for ep in eval_prefix], "epoch", file_name="{}.png".format(ef)),
            trigger=eval_trigger)

    # # snapshot.
    trainer.extend(
        training.extensions.snapshot_object(
            trainer.updater.get_optimizer('main').target.predictor,
            'model_snapshot_{.updater.epoch}.npz'),
        trigger=(settings["epoch_per_cycle"], 'epoch'))

    # trainer.extend(
    #     training.extensions.snapshot(filename='trainer_snapshot_epoch_{.updater.epoch}.npz'),
    #     trigger=(settings["max_epoch"], 'epoch'))

    # trainer.extend(
    #     training.extensions.snapshot_object(
    #         trainer.updater.get_optimizer('main').target.predictor,
    #         'model_snapshot_{.updater.epoch}.npz'),
    #     trigger=training.triggers.MinValueTrigger(
    #         "val/main/{}".format(settings["eval_functions"][0]), (1, "epoch")))

    return trainer
