# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

import os
import json

from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from easyai.data.datasets.caption_datasets import (
    CaptionDataset,
    CaptionInstructDataset,
    CaptionEvalDataset,
)

COCOCapDataset = CaptionDataset
COCOCapInstructDataset = CaptionInstructDataset


class COCOCapEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split("/")[-1].strip(".jpg").split("_")[-1]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }


class NoCapsEvalDataset(CaptionEvalDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["img_id"]

        return {
            "image": image,
            "image_id": img_id,
            "instance_id": ann["instance_id"],
        }
