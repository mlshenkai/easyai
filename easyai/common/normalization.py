# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/23 4:52 PM
# @File: normalization
# @Email: mlshenkai@163.com
import torch.nn as nn
from torch import Tensor


class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        output = nn.functional.layer_norm(
            input.float(),
            normalized_shape=self.normalized_shape,
            weight=self.weight.float() if self.weight is not None else None,
            bias=self.bias.float() if self.bias is not None else None,
            eps=self.eps,
        )
        return output.type_as(input)
