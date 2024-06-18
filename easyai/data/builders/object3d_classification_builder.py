# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.data.builders.base_dataset_builder import (
    MultiModalDatasetBuilder,
)
from easyai.data.datasets.multimodal.object3d_classification_datasets import (
    ModelNetClassificationMultiModelDataset,
)


@registry.register_builder("modelnet40_cls")
class ModelNetClassificationBuilder(MultiModalDatasetBuilder):
    train_dataset_cls = ModelNetClassificationMultiModelDataset
    eval_dataset_cls = ModelNetClassificationMultiModelDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/modelnet40/defaults_cls.yaml",
    }
