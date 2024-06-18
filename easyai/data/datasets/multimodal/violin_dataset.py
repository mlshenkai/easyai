# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

import os
import random
from easyai.data.datasets.multimodal.base_dataset import BaseMultiModelDataset

from easyai.data.datasets.multimodal.multimodal_classification_datasets import (
    MultimodalClassificationMultiModelDataset,
)


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["video_path"],
                "sentence": ann["sentence"],
                "label": ann["label"],
                "video": sample["video"],
            }
        )


class ViolinVideoEntailmentDataset(MultimodalClassificationMultiModelDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.class_labels = self._build_class_labels()

    def _build_class_labels(self):
        return {"wrong": 0, "correct": 1}

    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["video_path"]
        video_path = os.path.join(self.vis_root, vname)

        try:
            video = self.vis_processor(
                video_path, start_sec=ann["start_time"], end_sec=ann["end_time"]
            )
        except:
            return None

        sentence = self.text_processor(ann["statement"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "video_path": vname,
            "sentence": sentence,
            "label": self.class_labels[ann["label"]],
            "image_id": ann["source"],
            "instance_id": ann["instance_id"],
        }


class ViolinVideoEntailmentInstructDataset(ViolinVideoEntailmentDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        templates = [
            "is it true that {}?",
            "is the satement {} contained in the video?",
            "is the statement {} entailed in the video?",
        ]
        if data != None:
            data["text_output"] = "yes" if data["label"] == "correct" else "no"
            data["text_input"] = random.choice(templates).format(data["sentence"])
        return data


class ViolinVideoCaptionMultiModelDataset(BaseMultiModelDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.annotation = [ann for ann in self.annotation if ann["label"] == "correct"]

    def __getitem__(self, index):
        ann = self.annotation[index]

        vname = ann["video_path"]
        video_path = os.path.join(self.vis_root, vname)

        try:
            video = self.vis_processor(
                video_path, start_sec=ann["start_time"], end_sec=ann["end_time"]
            )
        except:
            return None
        caption = self.text_processor(ann["statement"])

        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "video": video,
            "text_input": self.text_processor(caption),
            "image_id": ann["source"],
            "instance_id": ann["instance_id"],
        }


class ViolinVideoCaptionInstructDataset(ViolinVideoCaptionMultiModelDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data["text_output"] = data["text_input"]
            data["text_input"] = self.text_processor("")
        return data


class ViolinVideoCaptionEvalDataset(ViolinVideoCaptionMultiModelDataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            del data["text_input"]
        return data
