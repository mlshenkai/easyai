# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.data.builders.object3d_caption_builder import ObjaverseCaptionBuilder
from easyai.data.datasets.multimodal.object3d_qa_datasets import ObjaverseQADataset


@registry.register_builder("objaverse_mm_qa")
class ObjaverseQABuilder(ObjaverseCaptionBuilder):
    train_dataset_cls = ObjaverseQADataset
    eval_dataset_cls = ObjaverseQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/objaverse/defaults_mm_qa.yaml",
    }
