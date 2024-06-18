# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from easyai.common.registry import registry
from easyai.data.datasets.multimodal.subject_driven_t2i_dataset import (
    SubjectDrivenTextToImageDataset,
)
from easyai.data.builders.base_dataset_builder import BaseDatasetBuilder


@registry.register_builder("blip_diffusion_finetune")
class BlipDiffusionFinetuneBuilder(BaseDatasetBuilder):
    train_dataset_cls = SubjectDrivenTextToImageDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/blip_diffusion_datasets/defaults.yaml"
    }

    def _download_ann(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        dataset = self.train_dataset_cls(
            image_dir=build_info.images.storage,
            subject_text=build_info.subject_text,
            inp_image_processor=self.kw_processors["inp_vis_processor"],
            tgt_image_processor=self.kw_processors["tgt_vis_processor"],
            txt_processor=self.text_processors["eval"],
        )

        return {"train": dataset}
