# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/11 下午5:03
# @File: base_task_tf
# @Email: mlshenkai@163.com
from dataclasses import fields

from easyai.common.registry import registry


class BaseTask:
    def __init__(self, **kwargs):
        super().__init__()

        self.inst_id_key = "instance_id"

    @classmethod
    def setup_task(cls, **kwargs):
        return cls()

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls(**model_config.config)

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
