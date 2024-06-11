# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/6 下午3:24
# @File: trainner
# @Email: mlshenkai@163.com
import argparse

from easyai.models.blip2.blip2_qformer import Blip2QFormerCLM
from easyai.models.blip2.configuration_blip2 import Blip2Config
from transformers import TrainingArguments, DataCollatorForSeq2Seq
import easyai.tasks as tasks
from easyai.common.config import Config


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


def main():
    cfg = Config(parse_args())
    cfg.pretty_print()
    task = tasks.setup_task(cfg)
    train_args = task.build_training_args(cfg)
    model = task.build_model(cfg)
    dataset = task.build_datasets(cfg)



    print(model)


if __name__ == "__main__":
    main()
