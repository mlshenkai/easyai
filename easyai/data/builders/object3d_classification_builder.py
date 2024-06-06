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
from easyai.data.datasets.object3d_classification_datasets import (
    ModelNetClassificationDataset,
)


@registry.register_builder("modelnet40_cls")
class ModelNetClassificationBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = ModelNetClassificationDataset
    eval_dataset_cls = ModelNetClassificationDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/modelnet40/defaults_cls.yaml",
    }
