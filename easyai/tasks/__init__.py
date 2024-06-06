# -*- coding: utf-8 -*-
# @Author: watcher
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.tasks.base_task import BaseTask
from easyai.tasks.captioning import CaptionTask
from easyai.tasks.image_text_pretrain import ImageTextPretrainTask
from easyai.tasks.multimodal_classification import (
    MultimodalClassificationTask,
)
from easyai.tasks.retrieval import RetrievalTask
from easyai.tasks.vqa import VQATask, GQATask, AOKVQATask, DisCRNTask
from easyai.tasks.vqa_reading_comprehension import VQARCTask, GQARCTask
from easyai.tasks.dialogue import DialogueTask
from easyai.tasks.text_to_image_generation import TextToImageGenerationTask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTask",
    "AOKVQATask",
    "RetrievalTask",
    "CaptionTask",
    "VQATask",
    "GQATask",
    "VQARCTask",
    "GQARCTask",
    "MultimodalClassificationTask",
    # "VideoQATask",
    # "VisualEntailmentTask",
    "ImageTextPretrainTask",
    "DialogueTask",
    "TextToImageGenerationTask",
    "DisCRNTask",
]
