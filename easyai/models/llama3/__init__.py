# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/13 下午2:31
# @File: __init__.py
# @Email: mlshenkai@163.com
import transformers
multiple_of = 256
hidden_dim=int(2 * 16384 / 3)
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
print(hidden_dim)