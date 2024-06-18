# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/5 下午4:06
# @File: base_blip2_model
# @Email: mlshenkai@163.com
from abc import ABC

from easyai.models import BaseModel


class Blip2BaseModel(BaseModel, ABC):
    def __init__(self):
        super().__init__()
