# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.data.builders.audio_caption_builder import AudioCapBuilder
from easyai.data.datasets.audio_qa_datasets import AudioCapsQADataset, ClothoQADataset


@registry.register_builder("audiocaps_mm_qa")
class AudioCapsQABuilder(AudioCapBuilder):
    train_dataset_cls = AudioCapsQADataset
    eval_dataset_cls = AudioCapsQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/audiocaps/defaults_mm_qa.yaml",
    }


@registry.register_builder("clotho_qa")
class ClothoQABuilder(AudioCapBuilder):
    train_dataset_cls = ClothoQADataset
    eval_dataset_cls = ClothoQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/clotho/defaults_mm_qa.yaml",
    }
