# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/6 上午10:28
# @File: blip2_qformer_config
# @Email: mlshenkai@163.com
from easyai.common.config import Config
from omegaconf import OmegaConf

args = OmegaConf.create(
    {
        "cfg_path": "/code-online/code/easy_ai/easyai/configs/models/blip2/blip2_qformer.yaml",
        "options": None,
    }
)
print(args.cfg_path)

config = Config(args)
