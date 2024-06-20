# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/13 下午2:39
# @File: configuration_llama3
# @Email: mlshenkai@163.com
from dataclasses import dataclass
from transformers import PretrainedConfig
from typing import Optional


@dataclass
class Llama3ModelConfig(PretrainedConfig):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048
    max_gen_len: int = 8016
