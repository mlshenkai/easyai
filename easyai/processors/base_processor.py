# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/27 10:58 AM
# @File: base_processor
# @Email: mlshenkai@163.com
from omegaconf import OmegaConf


class BaseProcessor:
    def __init__(self, *args, **kwargs):
        self.transform = lambda x: x

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        return cls()

    def build(self, **kwargs):
        cfg = OmegaConf.create(kwargs)
        return self.from_config(cfg)
