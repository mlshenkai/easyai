# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

import os

from PIL import Image

from easyai.data.datasets.multimodal.multimodal_classification_datasets import (
    MultimodalClassificationMultiModelDataset,
)
from easyai.data.datasets.multimodal.base_dataset import BaseMultiModelDataset


class VSRClassificationDataset(MultimodalClassificationMultiModelDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.class_labels = self._build_class_labels()
        self.classnames = ["no", "yes"]

    def _build_class_labels(self):
        return {"no": 0, "yes": 1}

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split(".")[0]

        return {
            "image": image,
            "image_id": img_id,
            "text_input": ann["caption"],
            "label": ann["label"],
            "instance_id": ann["instance_id"],
        }


class VSRClassificationInstructDataset(VSRClassificationDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data["answer"] = ["yes", "true"] if data["label"] == 1 else ["no", "false"]
            data["text_output"] = "yes" if data["label"] == 1 else "no"
        return data


class VSRCaptionMultiModelDataset(BaseMultiModelDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.annotation = [ann for ann in self.annotation if ann["label"] == 1]

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)

        img_id = ann["image"].split(".")[0]

        return {
            "image": image,
            "image_id": img_id,
            "text_input": ann["caption"],
        }


class VSRCaptionInstructDataset(VSRCaptionMultiModelDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data["text_output"] = data["text_input"]
            data["text_input"] = self.text_processor("")
        return data


class VSRCaptionEvalDataset(VSRCaptionMultiModelDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data
