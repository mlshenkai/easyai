# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

import os
from collections import OrderedDict

from easyai.data.datasets.multimodal.base_dataset import BaseMultiModelDataset
from PIL import Image


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )


class ImageTextPairMultiModelDataset(BaseMultiModelDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            return None

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {"image": image, "text_input": caption}


class ImageTextPairInstructDataset(ImageTextPairMultiModelDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data["text_output"] = data["text_input"]
            data["text_input"] = self.text_processor("")
        return data