# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:42 PM
# @File: __init__.py
# @Email: mlshenkai@163.com
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from easyai.configs import get_train_args

sys.argv = [
    "cli",
    "train",
    "--stage",
    "sft",
    "--do_train",
    "True",
    "--model_name_or_path",
    "/code",
    "--output_dir",
    "/code/logs",
    "--template",
    "default",
]


def args_test():
    print(sys.argv.pop(1))
    # args = " ".join(sys.argv[1:])
    # print(args)

    print(get_train_args())


if __name__ == "__main__":
    args_test()
