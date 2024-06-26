# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/5 下午2:38
# @File: base_model
# @Email: mlshenkai@163.com
import torch.nn as nn
from omegaconf import DictConfig
from abc import ABC
from easyai.common import get_abs_path


class BaseModel(nn.Module, ABC):
    PRETRAINED_MODEL_CONFIG_DICT = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    @classmethod
    def build_model_from_config(cls, config):
        if isinstance(config, (dict, DictConfig)):
            config = config["config"]["config"]
            config = cls.config_class(**config)
        return cls.from_pretrained(config.name_or_path, config=config)
