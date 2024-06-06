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
from easyai.data.datasets.nlvr_datasets import NLVRDataset, NLVREvalDataset
from easyai.data.datasets.snli_ve_datasets import (
    SNLIVisualEntialmentDataset,
    SNLIVisualEntialmentInstructDataset,
)
from easyai.data.datasets.violin_dataset import (
    ViolinVideoEntailmentDataset,
    ViolinVideoEntailmentInstructDataset,
)
from easyai.data.datasets.vsr_datasets import (
    VSRClassificationDataset,
    VSRClassificationInstructDataset,
)
from easyai.data.datasets.audio_classification_datasets import ESC50


@registry.register_builder("violin_entailment")
class ViolinEntailmentBuilder(BaseDatasetBuilder):
    train_dataset_cls = ViolinVideoEntailmentDataset
    eval_dataset_cls = ViolinVideoEntailmentDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/violin/defaults_entail.yaml",
    }


@registry.register_builder("violin_entailment_instruct")
class ViolinEntailmentInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = ViolinVideoEntailmentInstructDataset
    eval_dataset_cls = ViolinVideoEntailmentInstructDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/violin/defaults_entail_instruct.yaml",
    }


@registry.register_builder("nlvr")
class NLVRBuilder(BaseDatasetBuilder):
    train_dataset_cls = NLVRDataset
    eval_dataset_cls = NLVREvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/nlvr/defaults.yaml"}


@registry.register_builder("snli_ve")
class SNLIVisualEntailmentBuilder(BaseDatasetBuilder):
    train_dataset_cls = SNLIVisualEntialmentDataset
    eval_dataset_cls = SNLIVisualEntialmentDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/snli_ve/defaults.yaml"}


@registry.register_builder("snli_ve_instruct")
class SNLIVisualEntailmentInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = SNLIVisualEntialmentInstructDataset
    eval_dataset_cls = SNLIVisualEntialmentInstructDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/snli_ve/defaults_instruct.yaml"}


@registry.register_builder("vsr_classification")
class VSRClassificationBuilder(BaseDatasetBuilder):
    train_dataset_cls = VSRClassificationDataset
    eval_dataset_cls = VSRClassificationDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vsr/defaults_classification.yaml"
    }


@registry.register_builder("vsr_classification_instruct")
class SNLIVisualEntailmentInstructBuilder(BaseDatasetBuilder):
    train_dataset_cls = VSRClassificationInstructDataset
    eval_dataset_cls = VSRClassificationInstructDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vsr/defaults_classification_instruct.yaml"
    }


@registry.register_builder("esc50_cls")
class ESC50ClassificationBuilder(MultiModalDatasetBuilder):
    eval_dataset_cls = ESC50

    DATASET_CONFIG_DICT = {"default": "configs/datasets/esc50/defaults_mm_cls.yaml"}
