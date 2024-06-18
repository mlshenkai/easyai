# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

import os
from collections import OrderedDict

from PIL import Image

from easyai.data.datasets.multimodal.dialogue_datasets import DialogueMultiModelDataset

import json


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "dialogue": ann["dialogue"],
                "image": sample["image"],
            }
        )


class VisDialDataset(DialogueMultiModelDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root
        self.annotation = []
        for ann_path in ann_paths:
            data = json.load(open(ann_path, "r"))["data"]
            dialogs = data["dialogs"]
            answers = data["answers"]
            questions = data["questions"]

            for dialog in dialogs:
                all_turns = [
                    {
                        "answer": answers[d["answer"]],
                        "question": questions[d["question"]],
                    }
                    for d in dialog["dialog"]
                ]
                for i in range(len(all_turns)):
                    dialogue_context = " ".join(
                        [f" q: {t['question']} a: {t['answer']}" for t in all_turns[:i]]
                    ).strip()
                    last_turn = all_turns[i]

                    question = last_turn["question"]
                    answer = last_turn["answer"]

                    dialog["dialog"] = dialogue_context
                    dialog["question"] = question
                    dialog["answer"] = answer

                    self.annotation.append(dialog)

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(
            self.vis_root,
            "VisualDialog_train2018",
            f"VisualDialog_train2018_" + str(ann["image_id"]).zfill(12) + ".jpg",
        )
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        return {
            "image": image,
            "dialog": self.text_processor(ann["dialog"]),
            "text_input": self.text_processor(ann["question"]),
            "image_id": self.img_ids[ann["image_id"]],
            "answer": ann["answer"],
        }


class VisDialInstructDataset(VisDialDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data["text_output"] = data["answer"]
        return data


class VisDialEvalDataset(VisDialDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(
            self.vis_root,
            "VisualDialog_val2018",
            "VisualDialog_val2018_" + str(ann["image_id"]).zfill(12) + ".jpg",
        )
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        return {
            "image": image,
            "dialog": self.text_processor(ann["dialog"]),
            "text_input": self.text_processor(ann["question"]),
            "image_id": self.img_ids[ann["image_id"]],
            "answer": ann["answer"],
        }
