# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.common.utils import get_cache_path
from easyai.data.builders.base_dataset_builder import (
    BaseDatasetBuilder,
    MultiModalDatasetBuilder,
)
from easyai.data.datasets.multimodal.video_vqa_datasets import (
    VideoQADataset,
    VideoQAInstructDataset,
)
from easyai.data.datasets.multimodal.music_avqa import MusicAVQAInstructDataset, MusicAVQAMultiModelDataset


class VideoQABuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoQADataset
    eval_dataset_cls = VideoQADataset

    def build(self):
        datasets = super().build()

        ans2label = self.config.build_info.annotations.get("ans2label")
        if ans2label is None:
            raise ValueError("ans2label is not specified in build_info.")

        ans2label = get_cache_path(ans2label.storage)

        for split in datasets:
            datasets[split]._build_class_labels(ans2label)

        return datasets


@registry.register_builder("msrvtt_qa")
class MSRVTTQABuilder(VideoQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_qa.yaml",
    }


@registry.register_builder("msvd_qa")
class MSVDQABuilder(VideoQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_qa.yaml",
    }


@registry.register_builder("msrvtt_qa_instruct")
class MSRVTTQAInstructBuilder(VideoQABuilder):
    train_dataset_cls = VideoQAInstructDataset
    eval_dataset_cls = VideoQAInstructDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msrvtt/defaults_qa_instruct.yaml",
    }


@registry.register_builder("msvd_qa_instruct")
class MSVDQAInstructBuilder(VideoQABuilder):
    train_dataset_cls = VideoQAInstructDataset
    eval_dataset_cls = VideoQAInstructDataset
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/msvd/defaults_qa_instruct.yaml",
    }


@registry.register_builder("musicavqa_mm")
class MusicAVQABuilder(MultiModalDatasetBuilder):
    train_dataset_cls = MusicAVQAMultiModelDataset
    eval_dataset_cls = MusicAVQAMultiModelDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/music_avqa/defaults_mm_qa.yaml"}


@registry.register_builder("musicavqa_mm_instruct")
class MusicAVQAInstructBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = MusicAVQAInstructDataset
    eval_dataset_cls = MusicAVQAInstructDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/music_avqa/defaults_mm_qa_instruct.yaml"
    }
