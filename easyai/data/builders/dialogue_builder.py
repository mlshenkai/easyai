# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.data.builders.base_dataset_builder import (
    BaseDatasetBuilder,
    MultiModalDatasetBuilder,
)
from easyai.data.datasets.multimodal.avsd_dialogue_datasets import (
    AVSDDialDataset,
    AVSDDialEvalDataset,
    AVSDDialInstructEvalDataset,
)
from easyai.data.datasets.multimodal.visdial_dialogue_datasets import (
    VisDialDataset,
    VisDialInstructDataset,
    VisDialEvalDataset,
)

from easyai.data.datasets.multimodal.yt8m_video_dialogue_datasets import YT8MDialMultiModelDataset
from easyai.data.datasets.multimodal.llava150k_dataset import LLaVA150KInstructMultiModelDataset


@registry.register_builder("avsd_dialogue")
class AVSDDialBuilder(BaseDatasetBuilder):
    train_dataset_cls = AVSDDialDataset
    eval_dataset_cls = AVSDDialEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/avsd/defaults_dial.yaml"}


@registry.register_builder("visdial")
class VisDialBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisDialDataset
    eval_dataset_cls = VisDialEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/visdial/defaults_dial.yaml"}


@registry.register_builder("visdial_instruct")
class VisDialInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = VisDialInstructDataset
    eval_dataset_cls = VisDialEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/visdial/defaults_dial_instruct.yaml"
    }


@registry.register_builder("avsd_mm_dialogue_instruct")
class AVSDDialInstructBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = AVSDDialInstructEvalDataset
    eval_dataset_cls = AVSDDialInstructEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/avsd/defaults_mm_dial_instruct.yaml"
    }


@registry.register_builder("llava150k_dialogue_instruct")
class LLaVA150kDialInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = LLaVA150KInstructMultiModelDataset
    eval_dataset_cls = LLaVA150KInstructMultiModelDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/llava150k/defaults_dial.yaml"}


@registry.register_builder("yt8m_mm_dialogue")
class YT8MDialBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = YT8MDialMultiModelDataset
    eval_dataset_cls = YT8MDialMultiModelDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/yt8m/defaults_mm_dial.yaml"}
