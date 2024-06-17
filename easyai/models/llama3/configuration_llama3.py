# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/14 下午4:13
# @File: configuration_llama3
# @Email: mlshenkai@163.com
from typing import Optional

from transformers import LlamaConfig

class Llama3Config(LlamaConfig):

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        multiple_of: int = 256,  # make SwiGLU hidden layer size multiple of large power of 2
        ffn_dim_multiplier: Optional[float] = None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=50000.0,
        rope_scaling=None,
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pretraining_tp=pretraining_tp,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            **kwargs,
        )
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier

    # dim: int = 4096
    # n_layers: int = 32
    # n_heads: int = 32
    # n_kv_heads: Optional[int] = None
    # vocab_size: int = -1
    # multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    # ffn_dim_multiplier: Optional[float] = None
    # norm_eps: float = 1e-5
    # rope_theta: float = 500000
    #
    # max_batch_size: int = 32
    # max_seq_len: int = 2048
    # max_gen_len: int = 8016
    #
    # def