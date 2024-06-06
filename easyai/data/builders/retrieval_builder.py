# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from easyai.data.builders.base_dataset_builder import BaseDatasetBuilder
from easyai.data.datasets.retrieval_datasets import (
    RetrievalDataset,
    RetrievalEvalDataset,
    VideoRetrievalDataset,
    VideoRetrievalEvalDataset,
)

from easyai.common.registry import registry


@registry.register_builder("msrvtt_retrieval")
class MSRVTTRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/msrvtt/defaults_ret.yaml"}


@registry.register_builder("didemo_retrieval")
class DiDeMoRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/didemo/defaults_ret.yaml"}


@registry.register_builder("coco_retrieval")
class COCORetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coco/defaults_ret.yaml"}


@registry.register_builder("flickr30k")
class Flickr30kBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/flickr30k/defaults.yaml"}
