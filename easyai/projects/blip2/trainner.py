# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/6 下午3:24
# @File: trainner
# @Email: mlshenkai@163.com
import argparse

from easyai.data.data_utils import concat_datasets, reorg_datasets_by_split
from easyai.data.datasets.base_dataset import ConcatDataset
from easyai.models.blip2.blip2_qformer import Blip2QFormerCLM
from easyai.models.blip2.configuration_blip2 import Blip2Config
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
import torch
from torch.utils.data import DataLoader, IterableDataset
import easyai.tasks as tasks
from easyai.common.config import Config
from easyai.tasks.base_task_tf import BaseTask


def parse_args():
    parser = argparse.ArgumentParser(description="Train Blip2 model")
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="/code-online/code/easy_ai/easyai/projects/blip2/pretrain_stage1.yaml",
        help="path to config file",
    )
    parser.add_argument("--options", nargs="+")

    args = parser.parse_args()
    return args


class CaptionCollator:
    def __call__(self, features):
        samples = [s for s in features if s is not None]
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


def main():
    cfg = Config(parse_args())
    cfg.pretty_print()
    task: BaseTask = tasks.setup_task(cfg)
    train_args = task.build_training_args(cfg)
    model = task.build_model(cfg)
    datasets = task.build_datasets(cfg)
    optimizer, lr_scheduler = task.build_optimizers(model, cfg)
    datasets = reorg_datasets_by_split(datasets)
    datasets = concat_datasets(datasets)
    split_names = sorted(datasets.keys())
    datasets = [datasets[split] for split in split_names]
    for dataset in datasets:
        if isinstance(dataset, ConcatDataset):
            data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=dataset.collater)
            for data in data_loader:
                print(data)
    print(model)


if __name__ == "__main__":
    main()
