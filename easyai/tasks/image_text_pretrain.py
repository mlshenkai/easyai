# -*- coding: utf-8 -*-
# @Author: watcher
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.tasks.base_task_tf import BaseTask


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTaskWithTrainer(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass
