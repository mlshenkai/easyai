# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

import json
from tqdm import tqdm

train_file = "./train.jsonl"
test_file = "./test.jsonl"

train_data = [json.loads(l.strip()) for l in open(train_file).readlines()]
test_data = [json.loads(l.strip()) for l in open(test_file).readlines()]

for d in tqdm(train_data):
    d["video_path"] = d["video_id"] + ".mp4"
    d["ts"] = [float(d["start"]), float(d["end"])]

for d in tqdm(test_data):
    d["video_path"] = d["video_id"] + ".mp4"
    d["ts"] = [float(d["start"]), float(d["end"])]

json.dump(train_data, open("train_easyai.json", "w"))
json.dump(test_data, open("test_easyai.json", "w"))
