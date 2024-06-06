# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/5 下午3:23
# @File: base_dataset
# @Email: mlshenkai@163.com
import json
import torch
import pandas as pd
from typing import Iterable, Optional, List

from torch.utils.data import Dataset, ConcatDataset as BaseConcatDataset


class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root
        self.annotation = []
        for ann_path in ann_paths:
            if any(ext in ann_path for ext in ["csv", "tsv"]):
                df = pd.read_csv(ann_path)
                self.annotation.extend(df.to_dict(orient="records"))

            elif "jsonl" in ann_path:
                with open(ann_path, "r") as f:
                    self.annotation.extend([json.loads(line) for line in f])

            else:
                with open(ann_path, "r") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, list):
                        self.annotation.extend(loaded)
                    elif isinstance(loaded, dict):
                        self.annotation.extend(
                            [
                                {"sample_id": k, **v}
                                if isinstance(v, dict)
                                else {"sample_id": k, "data": v}
                                for k, v in loaded.items()
                            ]
                        )

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return {}
        collated_dict = {}
        keys = samples[0].keys()  # Use the keys of the first sample as a reference
        for k in keys:
            values = [sample[k] for sample in samples]
            # If the value type for the key is torch.Tensor, stack them else return list
            collated_dict[k] = (
                torch.stack(values, dim=0)
                if isinstance(values[0], torch.Tensor)
                else values
            )
        return collated_dict
        # return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)


class ConcatDataset(BaseConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]):
        super().__init__(datasets)

    def collater(self, samples):
        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
