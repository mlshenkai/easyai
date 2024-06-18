# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/11 下午5:03
# @File: base_task_tf
# @Email: mlshenkai@163.com
from dataclasses import fields
import torch.nn as nn
import torch
from easyai.common.registry import registry
from easyai.models.base_model import BaseModel
from loguru import logger as logging


# @registry.register_task("base_task")
class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls: BaseModel = registry.get_model_class(model_config.arch)
        return model_cls.build_model_from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]
            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def build_training_args(self, cfg):
        from transformers import TrainingArguments

        dataclass_fields = {field.name for field in fields(TrainingArguments)}
        args_dict = {}
        for field_key in dataclass_fields:
            if cfg.run_cfg.get(field_key) is not None:
                args_dict[field_key] = cfg.run_cfg.get(field_key)
        return TrainingArguments(**args_dict)

    def build_optimizers(self, model: BaseModel, cfg):
        lr_scale = cfg.run_cfg.get("lr_layer_decay", 1)
        weight_decay = cfg.run_cfg.get("weight_decay", 0.05)
        optim_params = model.get_optimizer_params(weight_decay, lr_scale)

        num_parameters = 0
        for p_group in optim_params:
            for p in p_group["params"]:
                num_parameters += p.data.nelement()
        logging.info("number of trainable parameters: {}".format(num_parameters))

        beta2 = cfg.run_cfg.get("beta2", 0.999)

        optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(cfg.run_cfg.learning_rate),
            betas=(0.9, beta2),
        )

        lr_sched_cls = registry.get_lr_scheduler_class(cfg.run_cfg.lr_sched)

        # max_epoch = self.config.run_cfg.max_epoch
        max_epoch = cfg.run_cfg.num_train_epochs
        # min_lr = self.config.run_cfg.min_lr
        min_lr = cfg.run_cfg.min_lr
        # init_lr = self.config.run_cfg.init_lr
        init_lr = cfg.run_cfg.learning_rate

        # optional parameters
        decay_rate = cfg.run_cfg.get("lr_decay_rate", None)
        warmup_start_lr = cfg.run_cfg.get("warmup_lr", -1)
        warmup_steps = cfg.run_cfg.get("warmup_steps", 0)

        lr_scheduler = lr_sched_cls(
            optimizer=optimizer,
            max_epoch=max_epoch,
            min_lr=min_lr,
            init_lr=init_lr,
            decay_rate=decay_rate,
            warmup_start_lr=warmup_start_lr,
            warmup_steps=warmup_steps,
        )
        return optimizer, lr_scheduler
