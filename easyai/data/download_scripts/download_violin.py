# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

import json
import os

json_path = "./violin_annotation.json"

## convert annotations
all_json = json.load(open(json_path))
train_data = [v for v in all_json.values() if "split" in v and v["split"] == "train"]
test_data = [v for v in all_json.values() if "split" in v and v["split"] == "test"]

json.dump(train_data, open("train.json", "w"))
json.dump(test_data, open("test.json", "w"))
