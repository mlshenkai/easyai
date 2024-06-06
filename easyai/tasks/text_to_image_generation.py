# -*- coding: utf-8 -*-
# @Author: watcher
# @Email: mlshenkai@163.com

from easyai.tasks import BaseTask
from easyai.common.registry import registry


@registry.register_task("text-to-image-generation")
class TextToImageGenerationTask(BaseTask):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        return cls(cfg=run_cfg)
