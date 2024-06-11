# -*- coding: utf-8 -*-
# @Author: watcher
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.tasks.base_task_with_train import BaseTaskWithTrainer
from easyai.tasks.captioning import CaptionTaskWithTrainer
from easyai.tasks.image_text_pretrain import ImageTextPretrainTaskWithTrainer
from easyai.tasks.multimodal_classification import (
    MultimodalClassificationTaskWithTrainer,
)
from easyai.tasks.retrieval import RetrievalTaskWithTrainer
from easyai.tasks.vqa import VQATaskWithTrainer, GQATask, AOKVQATask, DisCRNTask
from easyai.tasks.vqa_reading_comprehension import VQARCTask, GQARCTask
from easyai.tasks.dialogue import DialogueTaskWithTrainer
from easyai.tasks.text_to_image_generation import TextToImageGenerationTaskWithTrainer


def setup_task(cfg) -> BaseTaskWithTrainer:
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "BaseTaskWithTrainer",
    "AOKVQATask",
    "RetrievalTaskWithTrainer",
    "CaptionTaskWithTrainer",
    "VQATaskWithTrainer",
    "GQATask",
    "VQARCTask",
    "GQARCTask",
    "MultimodalClassificationTaskWithTrainer",
    # "VideoQATask",
    # "VisualEntailmentTask",
    "ImageTextPretrainTaskWithTrainer",
    "DialogueTaskWithTrainer",
    "TextToImageGenerationTaskWithTrainer",
    "DisCRNTask",
]
