# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/13 下午2:32
# @File: llama3_model.yaml
# @Email: mlshenkai@163.com
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import torch.nn as nn
import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from transformers import PreTrainedModel

from easyai.models.llama3.llama_fairscale.configuration_llama3 import Llama3ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.gamma


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    预计算 旋转位置编码
    :param dim:
    :param end:
    :param theta:
    :return:
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex 64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """

    :param freqs_cis: [num_head, head_size]
    :param x: tensor shape of [batch_size, num_head, seq_length, head_size]
    :return: [1, num_head, 1, head_size]
    """
    n_dim = x.ndim
    assert 0 <= 1 < n_dim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == n_dim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seqlen, n_kv_heads * n_rep, head_dim)
    )


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    :param xq: tensor shape of [batch_size, num_head, seq_length, head_size]
    :param xk: tensor shape of [batch_size, num_head, seq_length, head_size]
    :param freqs_cis: [num_head, head_size]
    :return:
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Attention(nn.Module):
    def __init__(self, config: Llama3ModelConfig):
        super().__init__()
        self.n_kv_heads = (
            config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        )
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.wq = ColumnParallelLinear(
            in_features=config.dim,
            out_features=config.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )

        self.wk = ColumnParallelLinear(
            in_features=config.dim,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )

        self.wv = ColumnParallelLinear(
            in_features=config.dim,
            out_features=self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )

        self.wo = RowParallelLinear(
            in_features=config.n_heads * self.head_dim,
            out_features=config.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                config.max_batch_size,
                config.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

        self.cache_v = torch.zeros(
            (
                config.max_batch_size,
                config.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[
            :bsz, : start_pos + seqlen
        ]  # (bsz, sql+cache_len, n_local_heads, head_dim)
        values = self.cache_v[
            :bsz, : start_pos + seqlen
        ]  # (bsz, sql+cache_len, n_local_heads, head_dim)

        keys = repeat_kv(
            keys, self.n_rep
        )  # (bsz, sql+cache_len, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bsz, sql+cache_len, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bsz, n_local_heads, sql, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_local_heads, sql+cache_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_local_heads, sql+cache_len, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores * mask  # (bsz, n_local_heads, sql, cache_len+sql)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bsz, b_local_heads, sql, head_dims)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            in_features=dim,
            out_features=hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )

        self.w2 = RowParallelLinear(
            in_features=hidden_dim,
            out_features=dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.w3 = ColumnParallelLinear(
            in_features=dim,
            out_features=hidden_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config: Llama3ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads

        self.attention = Attention(config)
        self.feed_forward = FeedForward(
            dim=self.dim,
            hidden_dim=4 * self.dim,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Llama3PreTrainedModel(PreTrainedModel):
    config_class = Llama3ModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True


class Llama3Model(Llama3PreTrainedModel):
    def __init__(self, config: Llama3ModelConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=config.dim,
            init_method=lambda x: x,
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id=layer_id, config=config))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        self.output = ColumnParallelLinear(
            in_features=config.dim,
            out_features=self.vocab_size,
            bias=False,
            init_method=lambda x: x,
        )

        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, config.max_seq_len * 2, config.rope_theta
        )

        self.gradient_checkpointing = False

    def get_input_embeddings(self) -> nn.Module:
        return self.tok_embeddings

    def set_input_embeddings(self, value):
        self.tok_embeddings = value

    # def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, pas):

    def forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
        _bsz, seqlen = x.shape
        h = self.tok_embeddings(x)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)

            mask = torch.stack(
                [torch.zeros((seqlen, start_pos), device=x.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output


class Llama3ForCausalLM(Llama3PreTrainedModel):
    def __init__(self, config: Llama3ModelConfig):
        super().__init__(config)
        self.config = config
        self.model = Llama3Model(config)
        self.vocab_size = config.vocab_size

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ):
        bsz = input_ids.shape[0]
        min_prompt_len = min(len(t) for t in input_ids)
        max_prompt_len = max(len(t) for t in input_ids)
        total_length = min(
            self.config.max_seq_len, max_prompt_len + self.config.max_gen_len
        )
        pad_id = self.model.tok_embeddings
