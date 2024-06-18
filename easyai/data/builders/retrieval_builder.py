# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from easyai.data.builders.base_dataset_builder import BaseDatasetBuilder
from easyai.data.datasets.multimodal.retrieval_datasets import (
    RetrievalMultiModelDataset,
    RetrievalEvalMultiModelDataset,
    VideoRetrievalMultiModelDataset,
    VideoRetrievalEvalMultiModelDataset,
)

from easyai.common.registry import registry


@registry.register_builder("msrvtt_retrieval")
class MSRVTTRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalMultiModelDataset
    eval_dataset_cls = VideoRetrievalEvalMultiModelDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/msrvtt/defaults_ret.yaml"}


@registry.register_builder("didemo_retrieval")
class DiDeMoRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalMultiModelDataset
    eval_dataset_cls = VideoRetrievalEvalMultiModelDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/didemo/defaults_ret.yaml"}


@registry.register_builder("coco_retrieval")
class COCORetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalMultiModelDataset
    eval_dataset_cls = RetrievalEvalMultiModelDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coco/defaults_ret.yaml"}


@registry.register_builder("flickr30k")
class Flickr30kBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalMultiModelDataset
    eval_dataset_cls = RetrievalEvalMultiModelDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/flickr30k/defaults.yaml"}
