# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/24 11:58 AM
# @File: test_utils
# @Email: mlshenkai@163.com
import cv2
import torch
import torch.nn as nn


def init_weights_with_constant(model: nn.Module, constant: float = 1.0) -> None:
    for n, p in model.named_parameters():
        nn.init.constant_(p, constant)
        # reduce the change to the tests
        for k in {
            "vision_projector.bias",
            "pooled_projection.bias",
            "output_projection.bias",
            "text_projector.bias",
        }:
            if n.endswith(k):
                nn.init.constant_(p, 0.0)
                break


if __name__ == "__main__":
    from easyai.models.blip2.blip2_qformer import Blip2QFormerCLM
    from easyai.processors.blip_processors import BlipImageEvalProcessor
    from PIL import Image
    from transformers import Blip2Config, Blip2VisionModel, Blip2Model

    config = Blip2Config()
    bert_pretrained_path = "/code-online/code/resources/models/bert-base-uncased"
    vision_pretrained_path = "/code-online/code/resources/models/blip2_vision_model.pth"
    qformer_model = Blip2QFormerCLM(
        config,
        tokenizer_pretrained_path=bert_pretrained_path,
        vision_pretrain_path=vision_pretrained_path,
    )
    init_weights_with_constant(qformer_model, 1.0)
    # state_dict = torch.load("/code-online/code/resources/models/eva_vit_g.pth", map_location="cpu")
    # miss_key = qformer_model.qformer_lm_head_model.qformer_model.load_state_dict(state_dict)
    vis_processor = BlipImageEvalProcessor(image_size=224)
    image = Image.open(
        "/code-online/code/easy_ai/examples/resources/imgs/sbu_caption.png"
    ).convert("RGB")
    image_tensor = vis_processor(image).unsqueeze(0)
    image_output = qformer_model.vision_model(image_tensor)
    print(image_output)

    print(qformer_model)
