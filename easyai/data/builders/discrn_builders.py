# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.data.builders.base_dataset_builder import (
    MultiModalDatasetBuilder,
)
from easyai.data.datasets.multimodal.discriminatory_reasoning_datasets import DisCRnMultiModelDataset


@registry.register_builder("image_pc_discrn")
class DiscrnImagePcBuilder(MultiModalDatasetBuilder):
    eval_dataset_cls = DisCRnMultiModelDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/discriminatory_reasoning/defaults_mm_image_pc.yaml",
    }


@registry.register_builder("audio_video_discrn")
class DiscrnAudioVideoBuilder(MultiModalDatasetBuilder):
    eval_dataset_cls = DisCRnMultiModelDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/discriminatory_reasoning/defaults_mm_audio_video.yaml",
    }
