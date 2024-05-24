# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/21 4:34 PM
# @File: blip2_vision
# @Email: mlshenkai@163.com
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import Blip2PreTrainedModel, Blip2VisionConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling


class Blip2VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.embed_dim = config.hidden_size

        self.class_embeds = nn.Parameter(torch.randn(1, 1, self.embed_dim))

        self.patch_embeds = nn.Conv2d(
            in_channels=3,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embeds = nn.Parameter(
            torch.randn(1, self.num_positions, self.embed_dim)
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        """

        :param pixel_values: [batch_size, 3, image_size, image_size]
        :return:
        """

        batch_size = pixel_values.shape[0]

        target_dtype = self.patch_embeds.weight.dtype

        patch_embeds = self.patch_embeds(
            pixel_values.to(target_dtype)
        )  # [batch_size, embed_dim, patch_size, patch_size]
        patch_embeds = patch_embeds.flatten(2).transpose(
            2, 1
        )  # [batch_size, patch_size*patch_size, embed_dim]

        class_embeds = self.class_embeds.expand(batch_size, 1, -1)

        embeddings = torch.cat(
            [class_embeds, patch_embeds], dim=1
        )  # [batch_size, patch_size*patch_size+1, embed_dim]

        embeddings = embeddings + self.position_embeds[:, : embeddings.size(1), :].to(
            target_dtype
        )
        return embeddings


class Blip2MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.scale = self.head_dim**-0.5  # sqrt(head_dim)

        self.dropout = nn.Dropout(p=config.attention_dropout)

        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=False)

        if config.qkv_bias:
            q_bias = nn.Parameter(torch.zeros(self.embed_dim))
            v_bias = nn.Parameter(torch.zeros(self.embed_dim))
        else:
            q_bias = None
            v_bias = None

        if q_bias is not None:
            qkv_bias = torch.cat(
                [q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias]
            )
            self.qkv.bias = nn.Parameter(qkv_bias)

        self.projector = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return (
            tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_length, embed_dim = hidden_states.size()

        mixed_qkv = self.qkv(hidden_states)  # [batch_size, seq_length, 3*embed_dim]
        mixed_qkv = mixed_qkv.reshape(
            batch_size, seq_length, 3, self.num_heads, embed_dim // self.num_heads
        ).permute(
            2, 0, 3, 1, 4
        )  # [3, batch_size, num_heads, seq_length, head_dim]

        query_states, key_states, value_states = (
            mixed_qkv[0],
            mixed_qkv[1],
            mixed_qkv[2],
        )

        attention_scores = torch.matmul(
            query_states, key_states.transpose(-1, -2)
        )  # [batch_size, num_heads, seq_length, seq_length]

        attention_scores = attention_scores * self.scale

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        attention_probs = self.dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(
            attention_probs, value_states
        )  # [batch_size, num_heads, seq_length, head_dim]
        context_layer = context_layer.permute(
            0, 2, 1, 3
        )  # [batch_size, seq_length, num_heads, head_dim]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.embed_dim,
        )  # [batch_size, seq_length] + [embed_dim, ] = [batch_size, seq_length, embed_dim]
        context_layer = context_layer.reshape(
            new_context_layer_shape
        )  # [batch_size, seq_length, embed_size]

        output = self.projector(context_layer)
        outputs = (output, attention_probs) if output_attentions else (output,)

        return outputs


class Blip2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.activation_fn = ACT2FN[config.hidden_act]

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class Blip2EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = Blip2MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = Blip2MLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states, head_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        output = hidden_states + residual
        outputs = (output, attn_weights) if output_attentions else (output,)
        return outputs


class Blip2Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Blip2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False

    def forward(
        self,
        input_embeds,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
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

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = input_embeds
        for idx, encoder_layer_module in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*input):
                        return module(*input, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward, hidden_states, attention_mask
                )
            else:
                layer_outputs = encoder_layer_module(
                    hidden_states, attention_mask, output_attentions=output_attentions
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, encoder_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )


class Blip2VisionModel(Blip2PreTrainedModel):
    main_input_name = "pixel_values"
    config_class = Blip2VisionConfig

    def __init__(self, config: Blip2VisionConfig):
        super().__init__(config)
        self.config = config

        embed_dim = config.hidden_size
        self.embeddings = Blip2VisionEmbeddings(config)
        self.encoder = Blip2Encoder(config)
        self.post_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
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

        hidden_states = self.embeddings(pixel_values)  # [batch_size, 1+patch_size*patch_size, hidden_size]

        encoder_outputs = self.encoder(
            input_embeds=hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_states = encoder_outputs[0]  # [batch_size, 1+patch_size*patch_size, hidden_size]
        last_hidden_states = self.post_layer_norm(last_hidden_states)  # [batch_size, 1+patch_size*patch_size, hidden_size]

        pooled_output = last_hidden_states[:, 0, :]
        pooled_output = self.post_layer_norm(pooled_output)

        if not return_dict:
            return (last_hidden_states, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_states,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embeddings
