# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

import os

from PIL import Image

from easyai.data.datasets.multimodal.caption_datasets import (
    __DisplMixin,
)
from easyai.data.datasets.multimodal.base_dataset import BaseMultiModelDataset


class CapFiltCaptionMultiModelDataset(BaseMultiModelDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            ann["image_id"] = "".join(ann["image"].split(".")[:-1])
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None  # image does not exist

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {"image": image, "text_input": caption, "image_id": ann["image_id"]}


class CapFiltCaptionInstructDataset(CapFiltCaptionMultiModelDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data["text_output"] = data["text_input"]
            data["text_input"] = self.text_processor("")
        return data