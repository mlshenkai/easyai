# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

import os
import random

from PIL import Image

from easyai.data.datasets.multimodal.vqa_datasets import VQAMultiModelDataset


class VGVQADataset(VQAMultiModelDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answers = [ann["answer"]]
        # TODO this should be configured better
        weights = [1.0]

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class VGVQAInstructDataset(VGVQADataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data["text_output"] = random.choice(data["answers"])
        return data

    def collater(self, samples):
        data = super().collater(samples)
        data["text_output"] = data["answer"]
        return data
