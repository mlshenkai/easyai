# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:39 PM
# @File: __init__.py
# @Email: mlshenkai@163.com
import os
import sys
from omegaconf import OmegaConf
from easyai.common.registry import registry
from easyai.common.optims import *
from easyai.models import *
from easyai.data.processors import *
from easyai.data.datasets import *
from easyai.data.builders import *


root_dir = os.path.dirname(os.path.abspath(__file__))
default_cfg = OmegaConf.load(os.path.join(root_dir, "configs/default.yaml"))

registry.register_path("library_root", root_dir)
repo_root = os.path.join(root_dir, "..")
registry.register_path("repo_root", repo_root)
cache_root = os.path.join(repo_root, default_cfg.env.cache_root)
registry.register_path("cache_root", cache_root)

registry.register("MAX_INT", sys.maxsize)
registry.register("SPLIT_NAMES", ["train", "val", "test"])
