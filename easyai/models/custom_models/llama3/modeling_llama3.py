# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/14 下午3:43
# @File: modeling_llama3
# @Email: mlshenkai@163.com
import math

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import Optional, Tuple, List, Union

from loguru import logger
from omegaconf import DictConfig
from transformers import PreTrainedModel
from easyai.common.auto_constants import AutoModelEnum
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss

from easyai.common.dist_utils import is_dist_avail_and_initialized
from easyai.configs import ModelArguments
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
import fairscale.nn.model_parallel.initialize as fs_init
from easyai.common.registry import registry
from .configuration_llama3 import Llama3Config


# copied by transformers.models.bart.modeling_bart._make_causal_mask
def _mask_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    make causal mask used for bi-directional self-attention
    :param input_ids_shape:
    :param dtype:
    :param device:
    :param past_key_values_length:
    :return:
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )

    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


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


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


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


class LlamaMLP(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        if is_dist_avail_and_initialized():
            self.gate_proj = ColumnParallelLinear(
                in_features=self.hidden_size,
                out_features=self.intermediate_size,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )

            self.up_proj = ColumnParallelLinear(
                in_features=self.hidden_size,
                out_features=self.intermediate_size,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )

            self.down_proj = RowParallelLinear(
                in_features=self.intermediate_size,
                out_features=self.hidden_size,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
        else:
            self.gate_proj = nn.Linear(
                in_features=self.hidden_size,
                out_features=self.intermediate_size,
                bias=False,
            )

            self.up_proj = nn.Linear(
                in_features=self.hidden_size,
                out_features=self.intermediate_size,
                bias=False,
            )

            self.down_proj = nn.Linear(
                in_features=self.intermediate_size,
                out_features=self.hidden_size,
                bias=False,
            )
        self.act_fn = ACT2FN[self.config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.config = config
        if is_dist_avail_and_initialized():
            model_parallel_size = fs_init.get_model_parallel_world_size()
        else:
            model_parallel_size = 1
        self.num_key_value_heads = config.num_key_value_heads
        self.num_heads = config.num_attention_heads

        self.hidden_size = config.hidden_size

        self.num_local_heads = config.num_attention_heads // model_parallel_size
        self.num_local_key_value_heads = self.num_key_value_heads // model_parallel_size
        self.num_key_value_groups = self.num_heads // self.num_local_key_value_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.head_dim = config.hidden_size // self.num_heads
        if is_dist_avail_and_initialized():
            self.q_proj = ColumnParallelLinear(
                in_features=self.hidden_size,
                out_features=self.num_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )

            self.k_proj = ColumnParallelLinear(
                in_features=self.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )

            self.v_proj = ColumnParallelLinear(
                in_features=self.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )

            self.o_proj = RowParallelLinear(
                in_features=self.num_heads * self.head_dim,
                out_features=self.hidden_size,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
        else:
            self.q_proj = nn.Linear(
                in_features=self.hidden_size,
                out_features=self.num_heads * self.head_dim,
                bias=False,
            )

            self.k_proj = nn.Linear(
                in_features=self.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=False,
            )

            self.v_proj = nn.Linear(
                in_features=self.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=False,
            )

            self.o_proj = nn.Linear(
                in_features=self.num_heads * self.head_dim,
                out_features=self.hidden_size,
                bias=False,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        bsz, seqlen, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, seqlen, self.num_local_heads, self.head_dim
        )
        key_states = key_states.view(
            bsz, seqlen, self.num_local_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, seqlen, self.num_local_key_value_heads, self.head_dim
        )

        query_states, key_states = apply_rotary_emb(
            query_states, key_states, position_ids
        )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.device
        )
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: Llama3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_size, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Llama3PreTrainedModel(PreTrainedModel):
    config_class = Llama3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/llama3/llama3_model.yaml"
    }

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, RowParallelLinear, ColumnParallelLinear)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Embedding, VocabParallelEmbedding)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Llama3Model(Llama3PreTrainedModel):
    @classmethod
    def build_model_from_config(cls, config: Llama3Config or dict):
        if isinstance(config, (dict, DictConfig)):
            config = config["config"]["config"]
            config = Llama3Config(**config)
        return cls.from_pretrained(config.name_or_path, config=config)

    def __init__(self, config: Llama3Config or dict):
        if isinstance(config, (dict, DictConfig)):
            config = Llama3Config(**config)
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        if is_dist_avail_and_initialized():
            self.embed_tokens = VocabParallelEmbedding(
                num_embeddings=self.vocab_size,
                embedding_dim=config.hidden_size,
                padding_idx=self.padding_idx,
                init_method=lambda x: x,
            )
        else:
            self.embed_tokens = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=config.hidden_size,
                padding_idx=self.padding_idx,
            )

        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape,
        input_embeds,
        pask_key_values_length,
    ) -> torch.Tensor:
        """
        create causal attention mask
        [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        :param attention_mask:
        :param input_shape:
        :param input_embeds:
        :param pask_key_values_length:
        :return:
        """
        combined_attention_mask = None

        if input_shape[-1] > 1:
            combined_attention_mask = _mask_causal_mask(
                input_shape,
                input_embeds.dtype,
                device=input_embeds.device,
                past_key_values_length=pask_key_values_length,
            )

        if attention_mask is not None:
            expanded_attn_mask = _expand_mask(
                attention_mask,
                input_embeds.dtype,
                tgt_len=input_shape[-1].to(input_embeds.device),
            )
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


@registry.register_model("llama3-7b", Llama3Config, AutoModelEnum.MODEL_FOR_CAUSAL_LM)
class Llama3ModelForCausalLM(Llama3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Union[Llama3Config, dict]):
        if isinstance(config, dict):
            config = Llama3Config(**config)
        super().__init__(config)
        self.model = Llama3Model(config)
        self.vocab_size = config.vocab_size
        if is_dist_avail_and_initialized():
            self.lm_head = ColumnParallelLinear(
                in_features=config.hidden_size,
                out_features=self.vocab_size,
                bias=False,
                init_method=lambda x: x,
            )
        else:
            self.lm_head = nn.Linear(
                in_features=config.hidden_size,
                out_features=self.vocab_size,
                bias=False,
            )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module):
        self.lm_head = value

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx, **kwargs):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past


@registry.register_model(
    "llama3-7b", Llama3Config, AutoModelEnum.MODEL_FOR_SEQUENCE_CLASSIFICATION
)
class Llama3ForSequenceClassification(Llama3PreTrainedModel):
    def __init__(self, config: Union[Llama3Config, dict]):
        if isinstance(config, dict):
            config = Llama3Config(**config)
        super().__init__(config)

        self.num_labels = config.num_labels
        self.model = Llama3Model(config)
        self.score = ColumnParallelLinear(
            in_features=config.hidden_size,
            out_features=self.num_labels,
            bias=False,
            init_method=lambda x: x,
        )
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.model.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError(
                "Cannot handle batch sizes > 1 if no padding token is defined."
            )
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (
                    torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1
                ).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device), sequence_lengths
        ]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    pooled_logits.view(-1, self.num_labels), labels.view(-1)
                )
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


if __name__ == "__main__":
    import torch.distributed
    import os

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_SIZE"] = "1"

    # model_config = Llama3Config.from_pretrained("/code-online/modelscope/llama3-chinese-Instruct")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    model_parallel_size = 1
    if not fs_init.model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        fs_init.initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1234)
    Llama3Model.from_pretrained("/code-online/modelscope/llama3-chinese-Instruct")
